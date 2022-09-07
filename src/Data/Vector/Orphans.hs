{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -Wno-orphans #-}

module Data.Vector.Orphans
  ( Vector (V_ProductF),
    MVector (MV_ProductF),
  )
where

import Data.Functor.Product (Product (..))
import Data.Vector.Unboxed
import Data.Vector.Unboxed.Deriving (derivingUnbox)

derivingUnbox
  "ProductF"
  [t|forall f g a. (Unbox (f a), Unbox (g a)) => Product f g a -> (f a, g a)|]
  [|\(Pair a b) -> (a, b)|]
  [|uncurry Pair|]
