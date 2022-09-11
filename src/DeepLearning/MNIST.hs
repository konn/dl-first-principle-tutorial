{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

module DeepLearning.MNIST
  ( toMNISTInput,
    toDigitVector,
    inferDigit,
    predict,
    predicts,
    accuracy,
    MNISTParams (..),
    DigitVector (..),
    MNISTInput,
    train,
    module Data.Format.MNIST,
  )
where

import Control.Applicative (liftA2)
import qualified Control.Foldl as L
import Control.Monad.Trans.Reader (ReaderT (..))
import Control.Subcategory
import Control.Subcategory.Linear
import Data.Coerce (coerce)
import Data.Distributive
import Data.Foldable (maximumBy)
import Data.Format.MNIST
import Data.Function (on)
import Data.Functor.Rep
import Data.Generics.Labels ()
import Data.Ord (comparing)
import qualified Data.Vector.Generic as G
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Generic.Mutable as MG
import qualified Data.Vector.Unboxed as U
import Data.Word
import DeepLearning.NeuralNetowrk.Massiv
import GHC.Generics (Generic, Generic1)
import GHC.TypeNats (KnownNat, type (*))
import Generic.Data (Generically1 (..))

data DigitVector a = DigitVector
  { is0
    , is1
    , is2
    , is3
    , is4
    , is5
    , is6
    , is7
    , is8
    , is9 ::
      !a
  }
  deriving (Show, Eq, Ord, Generic, Generic1, Functor, Foldable, Traversable)
  deriving anyclass (HasSize, FromVec)
  deriving (Applicative) via Generically1 DigitVector

newtype instance U.Vector (DigitVector a) = Vector_DV (U.Vector a)

newtype instance U.MVector s (DigitVector a) = MVector_DV {getDVMVector :: U.MVector s a}

instance U.Unbox a => G.Vector U.Vector (DigitVector a) where
  basicUnsafeFreeze = fmap coerce . G.basicUnsafeFreeze @U.Vector @a . coerce
  {-# INLINE basicUnsafeFreeze #-}
  basicUnsafeThaw = fmap coerce . G.basicUnsafeThaw @U.Vector @a . coerce
  {-# INLINE basicUnsafeThaw #-}
  basicLength = coerce $ (`quot` 10) . U.length @a
  {-# INLINE basicLength #-}
  basicUnsafeSlice = coerce (G.basicUnsafeSlice @U.Vector @a `on` (10 *))
  {-# INLINE basicUnsafeSlice #-}
  basicUnsafeIndexM (Vector_DV v) i =
    DigitVector
      <$> G.basicUnsafeIndexM v (i * 10)
      <*> G.basicUnsafeIndexM v (i * 10 + 1)
      <*> G.basicUnsafeIndexM v (i * 10 + 2)
      <*> G.basicUnsafeIndexM v (i * 10 + 3)
      <*> G.basicUnsafeIndexM v (i * 10 + 4)
      <*> G.basicUnsafeIndexM v (i * 10 + 5)
      <*> G.basicUnsafeIndexM v (i * 10 + 6)
      <*> G.basicUnsafeIndexM v (i * 10 + 7)
      <*> G.basicUnsafeIndexM v (i * 10 + 8)
      <*> G.basicUnsafeIndexM v (i * 10 + 9)
  {-# INLINE basicUnsafeIndexM #-}

instance U.Unbox a => MG.MVector U.MVector (DigitVector a) where
  basicLength = coerce $ (`quot` 10) . MG.basicLength @U.MVector @a
  {-# INLINE basicLength #-}
  basicUnsafeSlice = coerce (MG.basicUnsafeSlice @U.MVector @a `on` (10 *))
  {-# INLINE basicUnsafeSlice #-}
  basicOverlaps = coerce $ MG.overlaps @U.MVector @a
  {-# INLINE basicOverlaps #-}
  basicUnsafeNew = fmap MVector_DV . MG.basicUnsafeNew . (10 *)
  {-# INLINE basicUnsafeNew #-}
  basicInitialize = MG.basicInitialize . getDVMVector
  {-# INLINE basicInitialize #-}
  basicUnsafeRead (MVector_DV raw) =
    runReaderT $
      DigitVector
        <$> ReaderT (MG.basicUnsafeRead raw . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 1) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 2) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 3) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 4) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 5) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 6) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 7) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 8) . (10 *))
        <*> ReaderT (MG.basicUnsafeRead raw . (+ 9) . (10 *))
  {-# INLINE basicUnsafeRead #-}
  basicUnsafeWrite (MVector_DV raw) i DigitVector {..} = do
    MG.basicUnsafeWrite raw (10 * i) is0
    MG.basicUnsafeWrite raw (10 * i + 1) is1
    MG.basicUnsafeWrite raw (10 * i + 2) is2
    MG.basicUnsafeWrite raw (10 * i + 3) is3
    MG.basicUnsafeWrite raw (10 * i + 4) is4
    MG.basicUnsafeWrite raw (10 * i + 5) is5
    MG.basicUnsafeWrite raw (10 * i + 6) is6
    MG.basicUnsafeWrite raw (10 * i + 7) is7
    MG.basicUnsafeWrite raw (10 * i + 8) is8
    MG.basicUnsafeWrite raw (10 * i + 9) is9
  {-# INLINE basicUnsafeWrite #-}

instance U.Unbox a => U.Unbox (DigitVector a)

instance Representable DigitVector where
  type Rep DigitVector = Digit
  tabulate f = DigitVector (f D0) (f D1) (f D2) (f D3) (f D4) (f D5) (f D6) (f D7) (f D8) (f D9)
  index DigitVector {..} D0 = is0
  index DigitVector {..} D1 = is1
  index DigitVector {..} D2 = is2
  index DigitVector {..} D3 = is3
  index DigitVector {..} D4 = is4
  index DigitVector {..} D5 = is5
  index DigitVector {..} D6 = is6
  index DigitVector {..} D7 = is7
  index DigitVector {..} D8 = is8
  index DigitVector {..} D9 = is9

instance Distributive DigitVector where
  distribute = distributeRep
  collect = collectRep
  {-# INLINE distribute #-}

deriving anyclass instance (VectorSpace a a) => VectorSpace a (DigitVector a)

type MNISTInput n = UMat n n Double

toMNISTInput :: UMat n n Word8 -> MNISTInput n
toMNISTInput = cmap ((/ 255) . realToFrac)

digits :: DigitVector Digit
digits = tabulate id

inferDigit :: Ord a => DigitVector a -> Digit
{-# INLINE inferDigit #-}
inferDigit =
  fst
    . maximumBy (comparing snd)
    . liftA2 (,) digits

predict :: KnownNat n => NeuralNetwork (n * n) ls 10 Double -> MNISTInput n -> Digit
{-# INLINE predict #-}
predict = fmap inferDigit . evalF

predicts :: KnownNat n => NeuralNetwork (n * n) ls 10 Double -> U.Vector (MNISTInput n) -> U.Vector Digit
{-# INLINE predicts #-}
predicts = fmap (U.map inferDigit) . evalBatchF

accuracy :: U.Vector Digit -> U.Vector Digit -> Double
accuracy =
  fmap (L.foldOver vectorTraverse L.mean)
    . U.zipWith (\l r -> if l == r then 1.0 else 0.0)

data MNISTParams = MNISTParams
  { timeStep :: !Double
  , dumpingFactor :: !Double
  , adamParams :: !(AdamParams Double)
  }
  deriving (Show, Eq, Ord, Generic)

toDigitVector :: Digit -> DigitVector Double
toDigitVector basis = tabulate $ \ix ->
  if ix == basis then 1.0 else 0.0

train ::
  forall n ls.
  KnownNat n =>
  MNISTParams ->
  U.Vector (MNISTInput n, DigitVector Double) ->
  NeuralNetwork (n * n) ls 10 Double ->
  NeuralNetwork (n * n) ls 10 Double
{-# INLINE train #-}
train MNISTParams {..} =
  trainAdamF
    timeStep
    dumpingFactor
    adamParams
    1
    crossEntropy
