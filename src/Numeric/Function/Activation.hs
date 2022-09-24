{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Numeric.Function.Activation (sigmoid, relu, softmax) where

import Control.Arrow
import Control.Subcategory
import Control.Subcategory.Linear (Mat, computeM', computeV', delayM', duplicateAsRows', sumCols')
import Data.Massiv.Array (Ix1, Ix2, Load, Manifest, NumericFloat)
import Data.Type.Natural
import Numeric.Backprop (BVar, Backprop, Reifies, W, liftOp1, op1)

sigmoid :: Floating x => x -> x
{-# INLINE sigmoid #-}
sigmoid x = recip $ 1 + exp (-x)

relu ::
  (Backprop (f a), CZip f, Reifies s W, Ord a, Dom f a, Num a) =>
  BVar s (f a) ->
  BVar s (f a)
relu =
  liftOp1 $
    op1 (cmap (max 0) &&& czipWith (\x d -> if x < 0 then 0 else d))

softmax ::
  forall s r m k a.
  ( Reifies s W
  , KnownNat m
  , KnownNat k
  , Load r Ix2 a
  , Load r Ix1 a
  , NumericFloat r a
  , Manifest r a
  ) =>
  BVar s (Mat r m k a) ->
  BVar s (Mat r m k a)
softmax us =
  let exps = exp us
   in computeM' $ delayM' exps / duplicateAsRows' (computeV' @r $ sumCols' exps)
