// RUN: not tutorial-opt %s --affine-full-unroll 2>&1 | FileCheck %s

func.func @fail_unroll(%n: index, %buffer: memref<4xi32>) -> i32 {
  %c0 = arith.constant 0 : i32
  %sum = affine.for %i = 0 to %n iter_args(%acc = %c0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %next = arith.addi %acc, %t : i32
    affine.yield %next : i32
  }
  return %sum : i32
}

// CHECK: unrolling failed