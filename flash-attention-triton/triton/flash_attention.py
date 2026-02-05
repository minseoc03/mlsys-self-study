import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # ------------------------------------------------------------
    # This inner kernel processes a *range* of key/value positions (lo..hi)
    # for a fixed query block (BLOCK_SIZE_Q queries).
    #
    # It implements the numerically-stable "online softmax" accumulation:
    #   - maintain running row-wise max  m_i
    #   - maintain running row-wise sum  l_i = sum(exp(scores - m_i))
    #   - maintain running output accumulator O_block = sum(P * V)
    #
    # At the end of the full K/V sweep, the caller normalizes:
    #   O_block / l_i
    # and stores logsumexp (m_i + log(l_i)) for backward.
    # ------------------------------------------------------------

    # range of values handled by this stage
    if STAGE == 1:
        # Causal attention: blocks strictly to the *left* of the diagonal
        # For query block at position block_index_q, valid keys are [0, block_index_q*BLOCK_SIZE_Q)
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Causal attention: the "diagonal" block, where masking transitions happen within the block.
        # This stage handles keys in [block_index_q*BLOCK_SIZE_Q, (block_index_q+1)*BLOCK_SIZE_Q)
        # and applies an elementwise causal mask within that tile.
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Non-causal attention: all keys are visible [0, SEQ_LEN)
        lo, hi = 0, SEQ_LEN

    # ------------------------------------------------------------
    # Adjust K/V block pointers so that the loop starts at "lo".
    # K is accessed as [HEAD_DIM, SEQ_LEN] block (transposed layout), so advance along seq dim.
    # V is accessed as [SEQ_LEN, HEAD_DIM], so advance along seq dim as well (row offset).
    # ------------------------------------------------------------
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        # (helps alignment/pipelining assumptions for better codegen)
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        # Load one K tile: shape (HEAD_DIM, BLOCK_SIZE_KV) due to transposed representation
        K_block = tl.load(K_block_ptr)
        # Dot: (BLOCK_SIZE_Q, HEAD_DIM) x (HEAD_DIM, BLOCK_SIZE_KV) -> (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            # ------------------------------------------------------------
            # Apply causal mask *within* the diagonal tile:
            #   query position offs_q[i] can only attend to key positions <= offs_q[i]
            # Here, keys are at positions (start_kv + offs_kv[j]).
            #
            # mask=True means "this key is *after* the query (invalid)" in this formulation,
            # then we push logits to a very negative value so exp -> 0.
            # ------------------------------------------------------------
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            # For masked-out positions we add -1e6 (approx -inf), unmasked uses 0.
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)

            # Running max update (row-wise) for online softmax
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            # Shift logits by the new max for numerical stability
            QK_block -= m_ij[:, None]
        else:
            # Non-diagonal tiles: no intra-tile masking needed.
            # Multiply by softmax_scale before max so m_ij matches scaled logits.
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            # Shift scaled logits by new max
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute exp(logits - max): stable softmax numerator
        P_block = tl.math.exp(QK_block)
        # Row-wise sum of exp for this tile
        l_ij = tl.sum(P_block, 1)

        # ------------------------------------------------------------
        # "alpha" rescales the *previous* accumulator to the new max:
        # If old max was m_i and new max is m_ij, then:
        #   exp(old_logits - m_i) needs to become exp(old_logits - m_ij)
        # which equals exp(old_logits - m_i) * exp(m_i - m_ij)
        # so alpha = exp(m_i - m_ij)
        # ------------------------------------------------------------
        alpha = tl.math.exp(m_i - m_ij)

        # Update running denominator: l_i = l_i * alpha + l_ij
        l_i = l_i * alpha + l_ij

        # Load V tile: shape (BLOCK_SIZE_KV, HEAD_DIM)
        V_block = tl.load(V_block_ptr)

        # Dot uses fp16 P for speed; accumulation is into fp32 O_block
        P_block = P_block.to(tl.float16)

        # Update output accumulator:
        #   O_new = O_old * alpha + P @ V
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        # Commit running max
        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # ------------------------------------------------------------
    # Forward kernel: computes attention for one (batch, head, q-block).
    #
    # Program IDs:
    #   pid0 = which query block along sequence (block_index_q)
    #   pid1 = which (batch, head) pair (index_batch_head)
    # Each program instance produces a tile of O of shape (BLOCK_SIZE_Q, HEAD_DIM).
    # ------------------------------------------------------------
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = tl.program_id(1)
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # This allows to get the (SEQ_LEN, HEAD_DIM) slab for Q/K/V by selecting batch + head
    # (we reuse Q strides for base offset; K/V use their own strides below)
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # ------------------------------------------------------------
    # Block pointers:
    #  - Q: (SEQ_LEN, HEAD_DIM), read a tile at rows [block_index_q*BLOCK_SIZE_Q : +BLOCK_SIZE_Q)
    #  - K: represented as (HEAD_DIM, SEQ_LEN) to enable dot(Q, K) without explicit transpose
    #  - V: (SEQ_LEN, HEAD_DIM), streamed along sequence
    #  - O: output tile for the same query rows
    # order=(1,0) means the fastest varying dimension is the last one in memory access pattern.
    # ------------------------------------------------------------
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: absolute token indices for the Q rows handled by this program
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: local offsets within a K/V tile
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: running max per query row (for online logsumexp / softmax stability)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: running sum per query row (denominator of softmax, in exp-shifted space)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # O_block: accumulator for output tile in fp32
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Load Q tile once into SRAM; reused across all K/V blocks
    Q_block = tl.load(Q_block_ptr)

    # Stage convention:
    #   STAGE=3  => causal attention (do left-of-diagonal + diagonal masked tile)
    #   STAGE=1  => non-causal attention (single full sweep)

    if STAGE == 1 or STAGE == 3:
        # For non-causal: this covers full range
        # For causal: this covers left-of-diagonal range (no intra-tile mask needed)
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,  # maps STAGE=3 -> inner STAGE=1, STAGE=1 -> inner STAGE=3
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # Causal diagonal tile: apply intra-tile causal mask
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # ------------------------------------------------------------
    # Epilogue:
    # Store logsumexp for backward:
    #   logsumexp = m_i + log(l_i)
    # Normalize output:
    #   O = O_block / l_i
    # ------------------------------------------------------------
    m_i += tl.math.log(l_i)  # logsumexp per query row
    O_block = O_block / l_i[:, None]

    # Write M (logsumexp) and O tile back to memory
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # ------------------------------------------------------------
    # Backward preprocess:
    # For each query position i, compute:
    #   D_i = sum_j dO_i[j] * O_i[j]  (dot over HEAD_DIM)
    # This quantity appears in the softmax backward formula:
    #   dS = P * (dP - D_i)
    # ------------------------------------------------------------
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    # Load a block of O: shape (BLOCK_SIZE_Q, HEAD_DIM)
    O_block = tl.load(
        O
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    )

    # Load a block of dO: shape (BLOCK_SIZE_Q, HEAD_DIM)
    dO_block = tl.load(
        dO
        + index_batch_head * HEAD_DIM * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ).to(tl.float32)

    # Compute D_i per query row (BLOCK_SIZE_Q,)
    D_block = tl.sum(dO_block * O_block, axis=1)

    # Store D for later use in dq/dk/dv kernels
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # ------------------------------------------------------------
    # Backward kernel for dQ:
    # Fix a query block (BLOCK_Q rows) and sweep through all KV blocks.
    #
    # Uses saved M = logsumexp per query (from forward) to reconstruct softmax probs:
    #   P = exp(softmax_scale * QK - M)
    # and D to apply stable softmax backward.
    # ------------------------------------------------------------
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    # Load Q and dO tiles for this query block
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    # Load per-query logsumexp (stored as m_i + log(l_i) in forward)
    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]  # (BLOCK_Q, 1) for broadcasting

    offs_kv = tl.arange(0, BLOCK_KV)

    # We access the K and V as transposed blocks:
    #   kT_ptrs points to shape (HEAD_DIM, BLOCK_KV) region
    #   vT_ptrs points to shape (HEAD_DIM, BLOCK_KV) region
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    # Load D_i per query row: (BLOCK_Q,)
    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        # Load K^T and V^T tiles for current KV block
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)

        # Compute scaled attention logits for this (Q block) x (KV block)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)

        # Reconstruct softmax probabilities using saved logsumexp M:
        #   P = exp(QK - M)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # Autoregressive masking: only allow keys <= query index.
            # Here offs_q are absolute query positions, offs_kv are absolute key positions for current KV tile.
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            # Masked positions contribute 0 probability/gradient.
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP = dO @ V^T
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)

        # Softmax backward: dS = P * (dP - D_i)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        # dQ = softmax_scale * dS @ K
        # NOTE: we multiply by softmax_scale here because logits were scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        # Increment pointers to next KV tile
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    # Store dQ for this query block
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    # ------------------------------------------------------------
    # Backward kernel for dK and dV:
    # Fix a KV block and sweep over all query blocks.
    #
    # Computes:
    #   dV += P^T @ dO
    #   dK += softmax_scale * dS^T @ Q
    # where:
    #   P^T = exp(softmax_scale * KQ^T - M) with masking if causal
    #   dS^T = P^T * (dP^T - D)
    # ------------------------------------------------------------
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    # Accumulators for this KV block
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # Load K and V tiles for this KV block once (reused across all Q blocks)
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # (BLOCK_KV, HEAD_DIM)
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )  # (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # Access Q as transposed blocks so we can compute K @ Q^T efficiently
    # qT_ptrs has shape (HEAD_DIM, BLOCK_Q) conceptually
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Iterate over all query blocks
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load Q^T tile: (HEAD_DIM, BLOCK_Q)
        qT_block = tl.load(qT_ptrs)

        # Query indices for this block (absolute positions)
        offs_q = curr_q + tl.arange(0, BLOCK_Q)

        # Load per-query logsumexp M (shape (BLOCK_Q,))
        m = tl.load(M + offs_q)

        # Compute logits for this KV block against this Q block:
        #   QK_T_block = softmax_scale * (K @ Q^T)
        # Shape: (BLOCK_KV, BLOCK_Q)
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)

        # Softmax probs transposed:
        #   P_T = exp(QK_T - m)
        # Here m broadcasts over KV dimension.
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking in transposed form:
            # For each key position offs_kv[i] and query offs_q[j], valid iff query >= key.
            mask_block = offs_q[None, :] >= offs_kv[:, None]
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        # Load dO tile: (BLOCK_Q, HEAD_DIM)
        dO_block = tl.load(dO_ptrs)

        # dV += P^T @ dO   -> (BLOCK_KV, BLOCK_Q) x (BLOCK_Q, HEAD_DIM)
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Load D_i for this query block (BLOCK_Q,)
        Di = tl.load(D + offs_q)

        # dP^T = V @ dO^T  -> (BLOCK_KV, HEAD_DIM) x (HEAD_DIM, BLOCK_Q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # dS^T = P^T * (dP^T - D)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # dK += softmax_scale * dS^T @ Q
        # Here qT_block is (HEAD_DIM, BLOCK_Q), so trans(qT_block) is (BLOCK_Q, HEAD_DIM)
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        # Move to next Q block
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # Write back dV and dK tiles for this KV block
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)


class TritonAttention(torch.autograd.Function):
    # ------------------------------------------------------------
    # PyTorch autograd bridge:
    # forward() launches Triton attention forward kernel and saves tensors needed for backward.
    # backward() launches preprocess + two Triton kernels for gradients (dK/dV and dQ).
    #
    # Note: this is a "reference-style" Triton FlashAttention-like implementation:
    # - forward uses online softmax accumulation to avoid materializing full attention matrix
    # - backward reconstructs P using saved logsumexp (M) and applies stable softmax gradients
    # ------------------------------------------------------------

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # Q, K, V shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        # This implementation assumes all head dims match
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        # Output tensor
        O = torch.empty_like(Q)

        # STAGE=3 means causal (two-phase: left + diagonal mask), else STAGE=1 non-causal
        stage = 3 if causal else 1

        # Triton launch grid:
        #   dim0: number of Q blocks along SEQ_LEN
        #   dim1: BATCH_SIZE * NUM_HEADS (each program handles one (batch, head))
        #   dim2: unused (set to 1)
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M stores logsumexp per query position for backward (float32 for stability)
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        # Launch forward kernel (autotuned over block sizes / warps / stages)
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            # Pass strides explicitly so kernel works with arbitrary contiguous layout
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        # Retrieve saved tensors from forward
        Q, K, V, O, M = ctx.saved_tensors

        # This implementation expects contiguous layouts with matching strides across tensors
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        # Allocate gradients
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]

        # Kernel launch tuning knobs for backward (fixed here)
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        # Preprocess computes D_i = sum(dO_i * O_i) per query, in blocks
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        # Main backward grid:
        #   pid0: block along SEQ_LEN (either KV-block or Q-block depending on kernel)
        #   pid1: unused here (set to 1)
        #   pid2: batch_head index
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

        stage = 3 if ctx.causal else 1

        # ------------------------------------------------------------
        # Compute dK and dV:
        # Fix KV blocks (size BLOCK_KV=BLOCK_SIZE_MACRO), sweep through Q blocks (BLOCK_Q=BLOCK_SIZE_MICRO)
        # ------------------------------------------------------------
        _attn_bwd_dk_dv[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # ------------------------------------------------------------
        # Compute dQ:
        # Fix Q blocks (size BLOCK_Q=BLOCK_SIZE_MACRO), sweep through KV blocks (BLOCK_KV=BLOCK_SIZE_MICRO)
        # (note the swapped micro/macro roles vs dK/dV kernel)
        # ------------------------------------------------------------
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # Return gradients for inputs; causal and softmax_scale are non-tensor args (None)
        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    # ------------------------------------------------------------
    # Correctness test:
    # Compare Triton kernel outputs/gradients against a dense PyTorch reference:
    #   P = softmax(QK^T) [with causal mask if enabled]
    #   O = P @ V
    # Then backprop with random dO and compare dQ/dK/dV.
    # ------------------------------------------------------------
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation (dense attention)
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        # Set masked logits to -inf so softmax gives 0 probability there
        P[:, :, MASK == 0] = float("-inf")
    # softmax in float32 for numerical stability, then cast to fp16 for matmul
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)

    # Backprop reference
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare (tolerances chosen for fp16 numerical differences)
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Two quick runs: causal and non-causal
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=4096, HEAD_DIM=64, causal=False)
    print("PASSED")
