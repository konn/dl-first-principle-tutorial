module Numeric.Function.Activation (sigmoid, relu) where

sigmoid :: Floating x => x -> x
{-# INLINE sigmoid #-}
sigmoid x = recip $ 1 + exp (-x)

relu :: (RealFloat x) => x -> x
relu = max 0
