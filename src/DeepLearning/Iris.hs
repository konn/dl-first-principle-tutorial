{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

-- Day 1
module DeepLearning.Iris
  ( IrisFeatures (..),
    IrisVector (..),
    IrisKind (..),
    forward,
    train,
    classifyIris,
  )
where

import Control.Lens (alaf, foldMapOf)
import Data.Csv (DefaultOrdered, FromNamedRecord, FromRecord)
import Data.Foldable (maximumBy)
import Data.Functor.Compose (Compose (..))
import Data.List (iterate')
import Data.Monoid (Sum (..))
import Data.Ord (comparing)
import Data.Reflection (Reifies)
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import Data.Vector.Unboxed.Deriving (derivingUnbox)
import GHC.Generics (Generic, Generic1)
import Generic.Data (Generically1 (..))
import Linear (Additive ((^-^)), Metric (quadrance), (!*), (*^))
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Numeric.AD.Mode.Reverse (auto, grad)
import Type.Reflection (Typeable)

data IrisFeatures a = IrisFeatures {sepalLength, sepalWidth, petalLength, petalWidth :: !a}
  deriving (Show, Eq, Ord, Generic, Generic1, Functor, Foldable, Traversable)
  deriving anyclass (FromRecord, FromNamedRecord, DefaultOrdered, Additive, Metric)
  deriving (Applicative) via Generically1 IrisFeatures

data IrisVector a = IrisVector {setosa, versicolour, virginica :: !a}
  deriving (Show, Eq, Ord, Generic, Generic1, Functor, Foldable, Traversable)
  deriving anyclass (FromRecord, FromNamedRecord, DefaultOrdered, Additive, Metric)
  deriving (Applicative) via Generically1 IrisVector

data IrisKind = Setosa | Versicolour | Virginica
  deriving (Show, Eq, Ord, Generic)

classifyIris :: IrisVector Double -> IrisKind
classifyIris iv =
  fst $
    maximumBy (comparing snd) $
      (,) <$> IrisVector Setosa Versicolour Virginica
        <*> iv

derivingUnbox
  "IrisFeatures"
  [t|forall a. U.Unbox a => IrisFeatures a -> (a, a, a, a)|]
  [|\IrisFeatures {..} -> (sepalLength, sepalWidth, petalLength, petalWidth)|]
  [|\(sepalLength, sepalWidth, petalLength, petalWidth) -> IrisFeatures {..}|]

derivingUnbox
  "IrisVector"
  [t|forall a. U.Unbox a => IrisVector a -> (a, a, a)|]
  [|\IrisVector {..} -> (setosa, versicolour, virginica)|]
  [|\(setosa, versicolour, virginica) -> IrisVector {..}|]

sigmoid :: Floating x => x -> x
{-# INLINE sigmoid #-}
sigmoid x = recip $ 1 + exp (-x)

forward ::
  (Foldable r, Additive r, Functor v, Floating a) =>
  v (r a) ->
  r a ->
  v a
{-# INLINE forward #-}
forward w x =
  let value = w !* x
      activated = sigmoid <$> value
   in activated

-- | A simple gradient desecent with a static learning rate
gradDescent ::
  (Additive f, Traversable f, Num a) =>
  -- | Learning rate
  a ->
  (forall s. (Reifies s Tape, Typeable s) => f (Reverse s a) -> Reverse s a) ->
  f a ->
  [f a]
{-# INLINE gradDescent #-}
gradDescent gamma f = iterate' $ \x -> x ^-^ gamma *^ grad f x

train ::
  -- | Learning rate
  Double ->
  -- | # of Iterations
  Int ->
  -- | Inputs and answer
  U.Vector (IrisFeatures Double, IrisVector Double) ->
  -- | Current matrix
  IrisVector (IrisFeatures Double) ->
  IrisVector (IrisFeatures Double)
{-# INLINE train #-}
train gamma n cases =
  getCompose . last . take n
    . gradDescent
      gamma
      ( \(Compose w) ->
          alaf
            Sum
            (foldMapOf vectorTraverse)
            ( \(x, y) ->
                let y' = forward w $ auto <$> x
                 in quadrance $ y' ^-^ (auto <$> y)
            )
            cases
      )
    . Compose
