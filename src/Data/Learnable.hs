{-# LANGUAGE ConstraintKinds #-}

module Data.Learnable where

import Data.Vector.Unboxed (Unbox)

type Learnable a = (Unbox a)
