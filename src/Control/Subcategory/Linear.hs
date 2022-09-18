{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Control.Subcategory.Linear (
  Vec (),
  unsafeToVec,
  unVec,
  type UVec,
  type Vec1,
  type Vec2,
  type Vec3,
  type Vec4,
  repeatVec,
  asColumn,
  asRow,

  -- ** Vector operations
  (^+^),
  (^-^),
  (*^),
  (^*),
  (+^),
  (^+),
  (-^),
  (^-),
  (^/),
  (^.^),
  normVec,
  quadranceVec,

  -- ** Matrix operations
  Mat (),
  unsafeToMat,
  unMat,
  SomeBatch (..),
  splitColAt,
  splitRowAt,
  UMat,
  rowAt,
  columnAt,
  duplicateAsRows,
  duplicateAsCols,
  repeatMat,
  (!+!),
  (!+),
  (!-!),
  (!-),
  (!.!),
  (!*!),
  (!/!),
  (!^),
  (!*),
  (*!),
  (*!!),
  (!!*),
  (!!/),
  quadranceMat,
  normMat,

  -- * Conversion between vectors
  HasSize (..),
  GHasSize (),
  FromVec (..),
  genericFromVec,
  genericSinkToVector,
  GFromVec (),
  computeV,
  delayV,
  computeM,
  delayM,
  sumRows,
  sumCols,
  dimVal,
  trans,
  duplicateAsCols',
  duplicateAsRows',
  (!*:),
  (!*!:),
  sumRows',
  sumCols',
  replicateMatA,
  generateMatA,
  generateMat,
  replicateVecA,
  generateVecA,
  generateVec,
  fromBatchData,
  withDataPair,
  fromRowMat,

  -- ** Re-exports
  module Numeric.VectorSpace,
) where

import Control.DeepSeq (NFData)
import Control.Monad.ST
import Control.Subcategory
import qualified Data.Bifunctor as Bi
import Data.Coerce
import Data.Function (on)
import Data.Functor.Product (Product)
import Data.Massiv.Array (Sz (..))
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array.Manifest.Vector as VM
import qualified Data.Massiv.Core.Operations as M
import Data.Persist
import Data.Proxy (Proxy)
import Data.Sized (Sized, unsized)
import Data.Strict (Pair (..), type (:!:))
import qualified Data.Strict as SP
import qualified Data.Strict as St
import Data.These (These (..))
import Data.Type.Natural
import qualified Data.Vector as V
import qualified Data.Vector.Generic as G
import qualified Data.Vector.Generic.Mutable as MG
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as MU
import GHC.Base
import GHC.Generics hiding (V1)
import qualified Linear
import Linear.Affine (Point (..))
import qualified Linear.V as LinearV
import Massiv.Persist
import Numeric.Backprop
import Numeric.VectorSpace
import Type.Reflection (Typeable)

infixl 6 ^+^, ^-^, ^+, +^, ^-, -^

computeV :: (M.Manifest r a) => Vec M.D n a -> Vec r n a
computeV = coerce M.computeP

delayV :: (M.Source r a) => Vec r n a -> Vec M.D n a
delayV = coerce M.delay

computeM :: (M.Manifest r a) => Mat M.D n m a -> Mat r n m a
computeM = coerce M.computeP

delayM :: (M.Source r a) => Mat r n m a -> Mat M.D n m a
delayM = coerce M.delay

(^+^) :: (M.Numeric r a) => Vec r n a -> Vec r n a -> Vec r n a
{-# INLINE (^+^) #-}
(^+^) = coerce (M.!+!)

(^-^) :: (M.Numeric r a) => Vec r n a -> Vec r n a -> Vec r n a
{-# INLINE (^-^) #-}
(^-^) = coerce (M.!-!)

(+^) :: M.Numeric r a => a -> Vec r n a -> Vec r n a
{-# INLINE (+^) #-}
(+^) = coerce (M.+.)

(^+) :: M.Numeric r a => Vec r n a -> a -> Vec r n a
{-# INLINE (^+) #-}
(^+) = coerce (M..+)

(-^) :: M.Numeric r a => a -> Vec r n a -> Vec r n a
{-# INLINE (-^) #-}
(-^) = coerce (M.-.)

(^-) :: M.Numeric r a => Vec r n a -> a -> Vec r n a
{-# INLINE (^-) #-}
(^-) = coerce (M..-)

infixl 7 *^, ^*, ^/, ^.^

(*^) :: M.Numeric r a => a -> Vec r n a -> Vec r n a
{-# INLINE (*^) #-}
(*^) = coerce (M.*.)

(^*) :: M.Numeric r a => Vec r n a -> a -> Vec r n a
{-# INLINE (^*) #-}
(^*) = coerce (M..*)

(^/) :: M.NumericFloat r a => Vec r n a -> a -> Vec r n a
{-# INLINE (^/) #-}
(^/) = coerce (M../)

-- | Pointwise multiplication
(^.^) :: (M.Numeric r a) => Vec r n a -> Vec r n a -> Vec r n a
{-# INLINE (^.^) #-}
(^.^) = coerce (M.!*!)

sumRows :: (M.Source r a, Num a) => Mat r n m a -> Vec M.D m a
sumRows = coerce $ M.foldlWithin M.Dim1 (+) 0

sumCols :: (M.Source r a, Num a) => Mat r n m a -> Vec M.D n a
sumCols = coerce $ M.foldlWithin M.Dim2 (+) 0

infixl 7 !*!:, !*:

(!*!:) ::
  ( M.Numeric r a
  , M.Load r M.Ix2 a
  , KnownNat l
  , KnownNat m
  , KnownNat n
  , Reifies s W
  , M.Manifest r a
  ) =>
  BVar s (Mat r l m a) ->
  BVar s (Mat r n l a) ->
  BVar s (Mat r n m a)
{-# INLINE (!*!:) #-}
(!*!:) = liftOp2 $
  op2 $ \x y ->
    ( x !*! y
    , \d ->
        ( d !*! computeM (trans y)
        , computeM (trans x) !*! d
        )
    )

(!*:) ::
  ( M.Numeric r a
  , M.Load r M.Ix2 a
  , M.Load r M.Ix1 a
  , KnownNat m
  , KnownNat n
  , Reifies s W
  , M.Manifest r a
  ) =>
  BVar s (Mat r n m a) ->
  BVar s (Vec r n a) ->
  BVar s (Vec r m a)
{-# INLINE (!*:) #-}
(!*:) = liftOp2 $
  op2 $ \mNM vN ->
    ( computeV $ mNM !* vN
    , \dzdy ->
        ( asColumn dzdy !*! asRow vN
        , computeV (computeM (trans mNM) !* dzdy)
        )
    )

duplicateAsCols' ::
  forall m r n a s.
  ( KnownNat m
  , M.Manifest r a
  , Reifies s W
  , KnownNat n
  , M.Load r M.Ix1 a
  , M.NumericFloat r a
  , M.Load r M.Ix2 a
  ) =>
  BVar s (Vec r n a) ->
  BVar s (Mat r m n a)
{-# INLINE duplicateAsCols' #-}
duplicateAsCols' = liftOp1 $
  case sNat @1 %<=? sNat @m of
    SFalse -> 0.0
    STrue ->
      op1 $ \x ->
        ( computeM $ duplicateAsCols @m @r @n @a x
        , computeV . sumRows
        )

duplicateAsRows' ::
  forall m r n a s.
  ( KnownNat m
  , M.Manifest r a
  , Reifies s W
  , KnownNat n
  , M.Load r M.Ix1 a
  , M.NumericFloat r a
  , M.Load r M.Ix2 a
  ) =>
  BVar s (Vec r n a) ->
  BVar s (Mat r n m a)
{-# INLINE duplicateAsRows' #-}
duplicateAsRows' = liftOp1 $
  case sNat @1 %<=? sNat @m of
    SFalse -> 0.0
    STrue ->
      op1 $ \x ->
        ( computeM $ duplicateAsRows @m @r @n @a x
        , computeV . sumCols
        )

sumRows' ::
  ( Reifies s W
  , M.Source r a
  , Num a
  , M.Numeric r a
  , KnownNat n
  , KnownNat m
  , M.Load r M.Ix2 a
  , M.Manifest r a
  ) =>
  BVar s (Mat r n m a) ->
  BVar s (Vec r m a)
sumRows' = liftOp1 $
  op1 $ \mat ->
    (computeV $ sumRows mat, computeM . duplicateAsCols)

sumCols' ::
  ( Reifies s W
  , M.Source r a
  , Num a
  , M.Numeric r a
  , KnownNat n
  , KnownNat m
  , M.Load r M.Ix2 a
  , M.Manifest r a
  ) =>
  BVar s (Mat r n m a) ->
  BVar s (Vec r n a)
sumCols' = liftOp1 $
  op1 $ \mat ->
    (computeV $ sumCols mat, computeM . duplicateAsRows)

{-
>>> arr = Vec @M.U @3 $ M.fromList M.Par [1,2,3]
>>> mat = duplicateAsRows @2 arr
>>> mat
Mat {runMat = Array D Par (Sz (2 :. 3))
  [ [ 1.0, 2.0, 3.0 ]
  , [ 1.0, 2.0, 3.0 ]
  ]}

>>> sumRows mat
Vec {runVec = Array D Par (Sz1 2)
  [ 6.0, 6.0 ]}

>>> sumCols mat
Vec {runVec = Array D Par (Sz1 3)
  [ 2.0, 4.0, 6.0 ]}

 -}
instance (M.Numeric r a, KnownNat n, M.Load r M.Ix1 a) => Backprop (Vec r n a) where
  zero = const 0
  {-# INLINE zero #-}
  one = const 1
  {-# INLINE one #-}
  add = (^+^)
  {-# INLINE add #-}

instance
  (M.Numeric r a, KnownNat n, KnownNat m, M.Load r M.Ix2 a) =>
  Backprop (Mat r n m a)
  where
  zero = const 0
  {-# INLINE zero #-}
  one = const 1
  {-# INLINE one #-}
  add = (!+!)
  {-# INLINE add #-}

instance CFunctor (Vec r n) where
  cmap = coerce $ fmap M.computeP . M.map
  {-# INLINE cmap #-}

instance CSemialign (Vec r n) where
  calignWith f = coerce $ fmap M.computeP . M.zipWith (fmap f . These)
  {-# INLINE calignWith #-}

instance CZip (Vec r n) where
  czipWith = coerce $ fmap (fmap M.computeP) . M.zipWith
  {-# INLINE czipWith #-}

instance KnownNat n => CRepeat (Vec r n) where
  crepeat = repeatVec
  {-# INLINE crepeat #-}

instance KnownNat n => CPointed (Vec r n) where
  cpure = repeatVec
  {-# INLINE cpure #-}

repeatVec :: forall n r a. (KnownNat n, M.Load r M.Ix1 a) => a -> Vec r n a
{-# INLINE repeatVec #-}
repeatVec = Vec . M.replicate M.Par (M.Sz1 $ dimVal @n)

repeatMat :: forall n m r a. (KnownNat n, KnownNat m, M.Load r M.Ix2 a) => a -> Mat r n m a
{-# INLINE repeatMat #-}
repeatMat = Mat . M.replicate M.Par (M.Sz2 (dimVal @m) (dimVal @n))

instance (M.Numeric r a, KnownNat n, M.Load r M.Ix1 a) => Num (Vec r n a) where
  (+) = (^+^)
  {-# INLINE (+) #-}
  (-) = (^-^)
  {-# INLINE (-) #-}
  (*) = (^.^)
  {-# INLINE (*) #-}
  abs = coerce M.absA
  {-# INLINE abs #-}
  signum = coerce M.signumA
  {-# INLINE signum #-}
  fromInteger = repeatVec . fromInteger
  {-# INLINE fromInteger #-}
  negate = coerce M.negateA
  {-# INLINE negate #-}

instance
  (KnownNat n, M.Load r M.Ix1 a, M.NumericFloat r a) =>
  Fractional (Vec r n a)
  where
  recip = coerce M.recipA
  {-# INLINE recip #-}
  fromRational = repeatVec . fromRational
  {-# INLINE fromRational #-}
  (/) = coerce (M.!/!)
  {-# INLINE (/) #-}

instance
  (KnownNat n, M.Load r M.Ix1 a, M.NumericFloat r a, M.Manifest r a) =>
  Floating (Vec r n a)
  where
  pi = repeatVec pi
  {-# INLINE pi #-}
  exp = coerce M.expA
  {-# INLINE exp #-}
  log = coerce M.logA
  {-# INLINE log #-}
  logBase = coerce $ fmap M.computeP . M.logBaseA
  {-# INLINE logBase #-}
  sin = coerce M.sinA
  {-# INLINE sin #-}
  cos = coerce M.cosA
  {-# INLINE cos #-}
  tan = coerce M.tanA
  {-# INLINE tan #-}
  asin = coerce M.asinA
  {-# INLINE asin #-}
  acos = coerce M.acosA
  {-# INLINE acos #-}
  atan = coerce M.atanA
  {-# INLINE atan #-}
  sinh = coerce M.sinhA
  {-# INLINE sinh #-}
  cosh = coerce M.coshA
  {-# INLINE cosh #-}
  tanh = coerce M.tanhA
  {-# INLINE tanh #-}
  asinh = coerce M.asinhA
  {-# INLINE asinh #-}
  acosh = coerce M.acoshA
  {-# INLINE acosh #-}
  atanh = coerce M.atanhA
  {-# INLINE atanh #-}

instance (M.Numeric r a, KnownNat n, KnownNat m, M.Load r M.Ix2 a) => Num (Mat r n m a) where
  (+) = (!+!)
  {-# INLINE (+) #-}
  (-) = (!-!)
  {-# INLINE (-) #-}
  (*) = (!.!)
  {-# INLINE (*) #-}
  abs = coerce M.absA
  {-# INLINE abs #-}
  signum = coerce M.signumA
  {-# INLINE signum #-}
  fromInteger = repeatMat . fromInteger
  {-# INLINE fromInteger #-}
  negate = coerce M.negateA
  {-# INLINE negate #-}

instance (M.NumericFloat r a, KnownNat n, KnownNat m, M.Load r M.Ix2 a) => Fractional (Mat r n m a) where
  recip = coerce M.recipA
  {-# INLINE recip #-}
  fromRational = repeatMat . fromRational
  {-# INLINE fromRational #-}
  (/) = coerce (M.!/!)
  {-# INLINE (/) #-}

instance
  (KnownNat n, KnownNat m, M.Load r M.Ix2 a, M.NumericFloat r a, M.Manifest r a) =>
  Floating (Mat r n m a)
  where
  pi = repeatMat pi
  {-# INLINE pi #-}
  exp = coerce M.expA
  {-# INLINE exp #-}
  log = coerce M.logA
  {-# INLINE log #-}
  logBase = coerce $ fmap M.computeP . M.logBaseA
  {-# INLINE logBase #-}
  sin = coerce M.sinA
  {-# INLINE sin #-}
  cos = coerce M.cosA
  {-# INLINE cos #-}
  tan = coerce M.tanA
  {-# INLINE tan #-}
  asin = coerce M.asinA
  {-# INLINE asin #-}
  acos = coerce M.acosA
  {-# INLINE acos #-}
  atan = coerce M.atanA
  {-# INLINE atan #-}
  sinh = coerce M.sinhA
  {-# INLINE sinh #-}
  cosh = coerce M.coshA
  {-# INLINE cosh #-}
  tanh = coerce M.tanhA
  {-# INLINE tanh #-}
  asinh = coerce M.asinhA
  {-# INLINE asinh #-}
  acosh = coerce M.acoshA
  {-# INLINE acosh #-}
  atanh = coerce M.atanhA
  {-# INLINE atanh #-}

normVec :: (M.FoldNumeric r a, M.Source r a, Floating a) => Vec r n a -> a
{-# INLINE normVec #-}
normVec = coerce M.normL2

quadranceVec :: (M.FoldNumeric r a) => Vec r n a -> a
{-# INLINE quadranceVec #-}
quadranceVec = coerce $ flip M.powerSumArray 2

normMat :: (M.FoldNumeric r a, M.Source r a, Floating a) => Mat r n m a -> a
{-# INLINE normMat #-}
normMat = coerce M.normL2

quadranceMat :: (M.FoldNumeric r a) => Mat r n m a -> a
{-# INLINE quadranceMat #-}
quadranceMat = coerce $ flip M.powerSumArray 2

deriving newtype instance (HasSize v) => HasSize (Point v)

deriving newtype instance (FromVec v) => FromVec (Point v)

instance KnownNat n => HasSize (Sized U.Vector n) where
  type Size (Sized U.Vector n) = n
  toVec = Vec . M.fromUnboxedVector M.Par . unsized
  {-# INLINE toVec #-}
  sinkToVector mv = U.copy mv . unsized
  {-# INLINE sinkToVector #-}

type Vec1 = UVec 1

type Vec2 = UVec 2

type Vec3 = UVec 3

type Vec4 = UVec 4

newtype Vec r n a = Vec {runVec :: M.Vector r a}
  deriving (Generic)

deriving instance Eq (M.Vector r a) => Eq (Vec r n a)

unsafeToVec :: forall n r a. M.Vector r a -> Vec r n a
{-# INLINE unsafeToVec #-}
unsafeToVec = Vec

deriving newtype instance
  NFData (M.Vector r a) =>
  NFData (Vec r n a)

{-# INLINE unVec #-}
unVec :: Vec r n a -> M.Array r M.Ix1 a
unVec = runVec

type UVec = Vec M.U

instance Constrained (Vec r n) where
  type
    Dom (Vec r n) a =
      ( M.Load r M.Ix1 a
      , M.Manifest r a
      )

instance KnownNat n => HasSize (Vec M.U n) where
  type Size (Vec M.U n) = n
  toVec = id
  {-# INLINE toVec #-}
  sinkToVector mv = U.copy mv . M.toUnboxedVector . runVec
  {-# INLINE sinkToVector #-}

instance (KnownNat n, KnownNat m) => HasSize (Mat M.U n m) where
  type Size (Mat M.U n m) = m * n
  toVec = Vec . M.resize' (Sz1 (dimVal @m * dimVal @n)) . runMat
  {-# INLINE toVec #-}
  sinkToVector mv = U.copy mv . M.toUnboxedVector . runMat
  {-# INLINE sinkToVector #-}

type UMat = Mat M.U

newtype Mat r (n :: Nat) (m :: Nat) a = Mat {runMat :: M.Array r M.Ix2 a}
  deriving (Generic, Generic1)

deriving instance Eq (M.Matrix r a) => Eq (Mat r n m a)

unsafeToMat :: forall n m r a. M.Matrix r a -> Mat r n m a
{-# INLINE unsafeToMat #-}
unsafeToMat = Mat

deriving newtype instance
  NFData (M.Matrix r a) =>
  NFData (Mat r n m a)

unMat :: Mat r n m a -> M.Array r M.Ix2 a
{-# INLINE unMat #-}
unMat = runMat

deriving instance Show (M.Array r M.Ix2 a) => Show (Mat r n m a)

deriving instance Show (M.Array r M.Ix1 a) => Show (Vec r n a)

instance Constrained (Mat r n m) where
  type Dom (Mat r n m) a = (M.Load r M.Ix2 a, M.Manifest r a)

instance CFunctor (Mat r n m) where
  cmap f (Mat arr) = Mat $ M.computeP $ M.map f arr
  {-# INLINE cmap #-}

instance (KnownNat n, KnownNat m) => CPointed (Mat r n m) where
  cpure = Mat . M.replicate M.Par (M.Sz2 (dimVal @m) (dimVal @n))
  {-# INLINE cpure #-}

instance (KnownNat n, KnownNat m) => CSemialign (Mat r n m) where
  calignWith f =
    coerce $
      fmap (M.computeP @r) . M.zipWith (fmap f . These)
  {-# INLINE calignWith #-}

instance (KnownNat n, KnownNat m) => CZip (Mat r n m) where
  czipWith f =
    coerce $
      fmap (M.computeP @r) . M.zipWith f
  {-# INLINE czipWith #-}

instance (KnownNat n, KnownNat m) => CRepeat (Mat r n m) where
  crepeat = cpure
  {-# INLINE crepeat #-}

instance (KnownNat n, KnownNat m) => CFoldable (Mat r n m) where
  cfoldMap f = M.foldMono f . runMat
  {-# INLINE cfoldMap #-}
  clength = const $ dimVal @m * dimVal @n
  {-# INLINE clength #-}

instance (KnownNat n, KnownNat m) => CTraversable (Mat r n m) where
  ctraverse f = fmap Mat . M.traverseA f . runMat
  {-# INLINE ctraverse #-}

infixl 6 !+!, !-!, !+, !-

infixl 7 *!!, !!*, !*!, !.!, !!/, !/!, !*, *!

infixr 8 !^

(!+!) :: forall n m r a. M.Numeric r a => Mat r n m a -> Mat r n m a -> Mat r n m a
{-# INLINE (!+!) #-}
(!+!) = coerce (M.!+!)

(!-!) :: forall n m r a. M.Numeric r a => Mat r n m a -> Mat r n m a -> Mat r n m a
{-# INLINE (!-!) #-}
(!-!) = coerce (M.!-!)

(!+) :: forall n m r a. M.Numeric r a => Mat r n m a -> a -> Mat r n m a
{-# INLINE (!+) #-}
(!+) = coerce (M..+)

(!-) :: forall n m r a. M.Numeric r a => Mat r n m a -> a -> Mat r n m a
{-# INLINE (!-) #-}
(!-) = coerce (M..-)

(*!!) :: M.Numeric r a => a -> Mat r n m a -> Mat r n ma a
{-# INLINE (*!!) #-}
(*!!) = coerce (M.*.)

(!!*) :: M.Numeric r a => Mat r n m a -> a -> Mat r n ma a
{-# INLINE (!!*) #-}
(!!*) = coerce (M..*)

(!!/) :: M.NumericFloat r a => Mat r n m a -> a -> Mat r n ma a
{-# INLINE (!!/) #-}
(!!/) = coerce (M../)

-- | Matrix multiplication
(!*!) ::
  forall r n l m a.
  (M.Manifest r a, M.Numeric r a) =>
  Mat r l m a ->
  Mat r n l a ->
  Mat r n m a
{-# INLINE (!*!) #-}
(!*!) = coerce (M.!><!)

(!/!) ::
  forall r n l m a.
  (M.NumericFloat r a) =>
  Mat r n l a ->
  Mat r l m a ->
  Mat r n m a
{-# INLINE (!/!) #-}
(!/!) = coerce (M.!/!)

(!^) ::
  forall r n l a.
  (M.Numeric r a) =>
  Mat r n l a ->
  Int ->
  Mat r n l a
{-# INLINE (!^) #-}
(!^) = coerce M.powerPointwise

-- | Matrix-vector multiplication
(!*) ::
  forall r n l a.
  (M.Source r a, M.Numeric r a) =>
  Mat r n l a ->
  Vec r n a ->
  Vec M.D l a
{-# INLINE (!*) #-}
(!*) = coerce (M.!><)

-- | Row vector-matrix multiplication
(*!) ::
  forall r n m a.
  (M.Manifest r a, M.Numeric r a) =>
  Vec r n a ->
  Mat r n m a ->
  Vec r m a
{-# INLINE (*!) #-}
(*!) = coerce (M.><!)

-- | Pointwise matrix-matrix mutlipliation
(!.!) ::
  forall r n m a.
  (M.Numeric r a) =>
  Mat r n m a ->
  Mat r n m a ->
  Mat r n m a
{-# INLINE (!.!) #-}
(!.!) = coerce (M.!*!)

class (KnownNat (Size f)) => HasSize (f :: Type -> Type) where
  type Size f :: Nat
  type Size f = GSize (Rep1 f)
  toVec :: U.Unbox a => f a -> Vec M.U (Size f) a
  {-# INLINE toVec #-}
  toVec xs = Vec $
    M.fromUnboxedVector M.Par $
      U.create $ do
        mv <- MU.new $ dimVal @(Size f)
        sinkToVector mv xs
        pure mv

  sinkToVector :: U.Unbox a => U.MVector s a -> f a -> ST s ()
  default sinkToVector :: (Generic1 f, GHasSize (Rep1 f), U.Unbox a) => U.MVector s a -> f a -> ST s ()
  {-# INLINE sinkToVector #-}
  sinkToVector = genericSinkToVector

-- {-# INLINE sinkToVector #-}
-- sinkToVector =

deriving anyclass instance
  (HasSize f, HasSize g) =>
  HasSize (Product f g)

deriving anyclass instance
  (FromVec f, FromVec g) =>
  FromVec (Product f g)

class HasSize f => FromVec f where
  fromVec :: U.Unbox a => Vec M.U (Size f) a -> f a
  default fromVec ::
    (Generic1 f, GFromVec (Rep1 f)) =>
    U.Unbox a =>
    Vec M.U (Size f) a ->
    f a
  {-# INLINE fromVec #-}
  fromVec = genericFromVec
  decodeFrom :: U.Unbox a => U.Vector a -> f a :!: U.Vector a
  default decodeFrom ::
    (Generic1 f, GFromVec (Rep1 f)) =>
    U.Unbox a =>
    U.Vector a ->
    f a :!: U.Vector a
  {-# INLINE decodeFrom #-}
  decodeFrom = genericDecodeFrom

deriving anyclass instance FromVec Linear.V1

deriving anyclass instance FromVec Linear.V2

deriving anyclass instance FromVec Linear.V3

deriving anyclass instance FromVec Linear.V4

duplicateAsRows :: forall m r n a. (KnownNat m, M.Manifest r a) => Vec r n a -> Mat M.D n m a
duplicateAsRows = Mat . M.expandWithin M.Dim2 (M.Sz1 $ dimVal @m) const . runVec

duplicateAsCols :: forall m r n a. (KnownNat m, M.Manifest r a) => Vec r n a -> Mat M.D m n a
duplicateAsCols = Mat . M.expandWithin M.Dim1 (M.Sz1 $ dimVal @m) const . runVec

genericFromVec :: (Generic1 f, GFromVec (Rep1 f), U.Unbox a) => Vec M.U (Size f) a -> f a
{-# INLINE genericFromVec #-}
genericFromVec = to1 . St.fst . gDecodeFrom . M.toUnboxedVector . runVec

genericDecodeFrom :: (Generic1 f, GFromVec (Rep1 f), U.Unbox a) => U.Vector a -> f a :!: U.Vector a
{-# INLINE genericDecodeFrom #-}
genericDecodeFrom = Bi.first to1 . gDecodeFrom

genericSinkToVector :: (Generic1 f, GHasSize (Rep1 f), U.Unbox a) => U.MVector s a -> f a -> ST s ()
{-# INLINE genericSinkToVector #-}
genericSinkToVector mv = gsinkToVector mv . from1

class (KnownNat (GSize f)) => GHasSize f where
  type GSize f :: Nat
  gsinkToVector :: U.Unbox a => U.MVector s a -> f a -> ST s ()

class GHasSize f => GFromVec f where
  gDecodeFrom :: U.Unbox a => U.Vector a -> f a :!: U.Vector a

instance GHasSize f => GHasSize (M1 i c f) where
  type GSize (M1 i c f) = GSize f
  gsinkToVector mv = gsinkToVector mv . unM1
  {-# INLINE gsinkToVector #-}

instance GFromVec f => GFromVec (M1 i c f) where
  gDecodeFrom = coerce $ gDecodeFrom @f
  {-# INLINE gDecodeFrom #-}

instance HasSize f => GHasSize (Rec1 f) where
  type GSize (Rec1 f) = Size f
  gsinkToVector mv = sinkToVector mv . unRec1
  {-# INLINE gsinkToVector #-}

instance FromVec f => GFromVec (Rec1 f) where
  gDecodeFrom = Bi.first Rec1 . decodeFrom
  {-# INLINE gDecodeFrom #-}

instance (GHasSize f, GHasSize g) => GHasSize (f :*: g) where
  type GSize (f :*: g) = GSize f + GSize g
  gsinkToVector mv (f :*: g) = do
    let (lh, rh) = MU.splitAt (dimVal @(GSize f)) mv
    gsinkToVector lh f
    gsinkToVector rh g
  {-# INLINE gsinkToVector #-}

instance (GFromVec f, GFromVec g) => GFromVec (f :*: g) where
  gDecodeFrom xs =
    let (fa :!: rest) = gDecodeFrom xs
        (ga :!: rest') = gDecodeFrom rest
     in (fa :*: ga) :!: rest'
  {-# INLINE gDecodeFrom #-}

instance GHasSize Par1 where
  type GSize Par1 = 1
  gsinkToVector mv = MU.write mv 0 . unPar1
  {-# INLINE gsinkToVector #-}

instance GFromVec Par1 where
  gDecodeFrom =
    coerce $ Bi.first U.head . St.toStrict . U.splitAt @a 1 ::
      forall a. U.Unbox a => U.Vector a -> Par1 a :!: U.Vector a
  {-# INLINE gDecodeFrom #-}

instance GHasSize (K1 i c) where
  type GSize (K1 i c) = 0
  gsinkToVector = mempty
  {-# INLINE gsinkToVector #-}

instance GHasSize U1 where
  type GSize U1 = 0
  gsinkToVector = mempty
  {-# INLINE gsinkToVector #-}

instance GFromVec U1 where
  gDecodeFrom = (U1 :!:)
  {-# INLINE gDecodeFrom #-}

deriving anyclass instance HasSize Linear.V0

deriving anyclass instance FromVec Linear.V0

deriving anyclass instance HasSize Linear.V1

deriving anyclass instance HasSize Linear.V2

deriving anyclass instance HasSize Linear.V3

deriving anyclass instance HasSize Linear.V4

instance KnownNat n => FromVec (UVec n) where
  fromVec = id
  {-# INLINE fromVec #-}
  decodeFrom =
    Bi.first (Vec . M.fromUnboxedVector M.Par)
      . SP.toStrict
      . U.splitAt (dimVal @n)
  {-# INLINE decodeFrom #-}

instance (KnownNat n, KnownNat m) => FromVec (UMat n m) where
  fromVec = Mat . M.resize' (Sz2 (dimVal @m) (dimVal @n)) . runVec
  {-# INLINE fromVec #-}
  decodeFrom = Bi.first fromVec . decodeFrom
  {-# INLINE decodeFrom #-}

instance KnownNat n => HasSize (LinearV.V n) where
  type Size (LinearV.V n) = n
  toVec = Vec . VM.fromVector' M.Par (M.Sz1 $ dimVal @n) . LinearV.toVector
  {-# INLINE toVec #-}
  sinkToVector mv (LinearV.V vec) = do
    U.copy mv $ G.convert vec

dimVal :: forall n. KnownNat n => Int
dimVal = fromIntegral $ natVal' @n proxy#

rowAt ::
  forall i m n r a.
  ( KnownNat i
  , i + 1 <= m
  , KnownNat n
  , M.Source r a
  ) =>
  Mat r n m a ->
  Vec M.D n a
{-# INLINE rowAt #-}
rowAt =
  Vec
    . M.resize' (M.Sz1 $ dimVal @n)
    . M.extract' (dimVal @i M.:. 0) (M.Sz2 1 $ dimVal @n)
    . runMat

columnAt ::
  forall i m n r a.
  ( KnownNat i
  , i + 1 <= n
  , KnownNat m
  , M.Source r a
  ) =>
  Mat r n m a ->
  Vec M.D m a
{-# INLINE columnAt #-}
columnAt =
  Vec
    . M.resize' (M.Sz1 $ dimVal @m)
    . M.extract' (0 M.:. dimVal @i) (M.Sz2 (dimVal @m) 1)
    . runMat

asColumn :: forall r n a. (M.Size r, KnownNat n) => Vec r n a -> Mat r 1 n a
{-# INLINE asColumn #-}
asColumn = coerce $ M.resize' (M.Sz2 (dimVal @n) 1)

asRow :: forall r n a. (M.Size r, KnownNat n) => Vec r n a -> Mat r n 1 a
{-# INLINE asRow #-}
asRow = coerce $ M.resize' (M.Sz2 1 (dimVal @n))

instance
  (M.NumericFloat r a, KnownNat n, M.Manifest r a, M.Load r M.Ix1 a) =>
  VectorSpace a (Vec r n a)
  where
  reps = cpure
  {-# INLINE reps #-}
  (.+) = (+^)
  {-# INLINE (.+) #-}
  (+.) = (^+)
  {-# INLINE (+.) #-}
  (.-) = (-^)
  {-# INLINE (.-) #-}
  (-.) = (^-)
  {-# INLINE (-.) #-}
  (.*) = (*^)
  {-# INLINE (.*) #-}
  (*.) = (^*)
  {-# INLINE (*.) #-}
  (/.) = (^/)
  {-# INLINE (/.) #-}

  (>.<) = coerce (M.!.!)
  {-# INLINE (>.<) #-}

  sumS = coerce M.sum
  {-# INLINE sumS #-}

instance
  (M.NumericFloat r a, KnownNat n, KnownNat m, M.Manifest r a, M.Load r M.Ix2 a) =>
  VectorSpace a (Mat r n m a)
  where
  reps = cpure
  {-# INLINE reps #-}
  (.+) = coerce (M.+.)
  {-# INLINE (.+) #-}
  (+.) = coerce (M..+)
  {-# INLINE (+.) #-}
  (.-) = coerce (M.-.)
  {-# INLINE (.-) #-}
  (-.) = coerce (M..-)
  {-# INLINE (-.) #-}
  (.*) = (*!!)
  {-# INLINE (.*) #-}
  (*.) = (!!*)
  {-# INLINE (*.) #-}
  (/.) = (!!/)
  {-# INLINE (/.) #-}
  sumS = coerce M.sum
  {-# INLINE sumS #-}
  (>.<) = coerce ((M.!.!) `on` M.resize' (M.Sz1 $ dimVal @n * dimVal @m))
  {-# INLINE (>.<) #-}

trans :: M.Source r a => Mat r n m a -> Mat M.D m n a
{-# INLINE trans #-}
trans = coerce M.transpose

replicateMatA ::
  forall n m a r f.
  (KnownNat n, KnownNat m, Applicative f, M.Manifest r a) =>
  f a ->
  f (Mat r n m a)
{-# INLINE replicateMatA #-}
replicateMatA =
  fmap Mat . M.makeArrayA (M.Sz2 (dimVal @m) (dimVal @n)) . const

generateMatA ::
  forall n m a r f.
  (KnownNat n, KnownNat m, Applicative f, M.Manifest r a) =>
  (M.Ix2 -> f a) ->
  f (Mat r n m a)
{-# INLINE generateMatA #-}
generateMatA =
  fmap Mat . M.makeArrayA (M.Sz2 (dimVal @m) (dimVal @n))

generateMat ::
  forall n m a r.
  (KnownNat n, KnownNat m, M.Manifest r a, M.Load r M.Ix2 a) =>
  (M.Ix2 -> a) ->
  Mat r n m a
{-# INLINE generateMat #-}
generateMat =
  Mat . M.makeArray M.Par (M.Sz2 (dimVal @m) (dimVal @n))

replicateVecA ::
  forall n a r f.
  (KnownNat n, Applicative f, M.Manifest r a) =>
  f a ->
  f (Vec r n a)
{-# INLINE replicateVecA #-}
replicateVecA =
  fmap Vec . M.makeArrayA (M.Sz1 (dimVal @n)) . const

generateVecA ::
  forall n a r f.
  (KnownNat n, Applicative f, M.Manifest r a) =>
  (M.Ix1 -> f a) ->
  f (Vec r n a)
{-# INLINE generateVecA #-}
generateVecA =
  fmap Vec . M.makeArrayA (M.Sz1 (dimVal @n))

generateVec ::
  forall n a r.
  (KnownNat n, M.Manifest r a, M.Load r M.Ix1 a) =>
  (Int -> a) ->
  Vec r n a
{-# INLINE generateVec #-}
generateVec =
  Vec . M.makeArray M.Par (M.Sz1 (dimVal @n))

data SomeBatch t a where
  MkSomeBatch :: (KnownNat m) => UMat m (Size t) a -> SomeBatch t a

deriving instance (Show a, U.Unbox a) => Show (SomeBatch t a)

fromBatchData ::
  forall t a v.
  (G.Vector v (t a), HasSize t, U.Unbox a) =>
  v (t a) ->
  SomeBatch t a
fromBatchData ts =
  case someNatVal $ fromIntegral $ G.length ts of
    SomeNat (_ :: Proxy m) ->
      MkSomeBatch @m $
        Mat $
          M.computeP $
            M.concat' 1 $
              V.map (runMat . asColumn . toVec) $
                G.convert ts

withDataPair ::
  forall t u a v r.
  ( G.Vector v (t a, u a)
  , G.Vector v (t a)
  , G.Vector v (u a)
  , Typeable v
  , U.Unbox (t a)
  , U.Unbox (u a)
  , HasSize t
  , HasSize u
  , U.Unbox a
  , M.Load (VM.ARepr v) M.Ix1 (t a)
  , M.Load (VM.ARepr v) M.Ix1 (u a)
  ) =>
  v (t a, u a) ->
  ( forall m.
    KnownNat m =>
    (UMat m (Size t) a, UMat m (Size u) a) ->
    r
  ) ->
  r
withDataPair dats act =
  let (ins, outs) = G.unzip dats
      m = G.length ins
      inMat =
        M.computeP $
          M.concat' 1 $
            M.map (runMat . asColumn . toVec) $
              VM.fromVector' @_ @_ @_ @M.U M.Par (Sz1 m) ins
      outMat =
        M.computeP $
          M.concat' 1 $
            M.map (runMat . asColumn . toVec) $
              VM.fromVector' @_ @_ @_ @M.U M.Par (Sz1 m) outs
   in case someNatVal $ fromIntegral m of
        SomeNat (_ :: Proxy m) ->
          act @m (Mat inMat, Mat outMat)

newtype instance U.Vector (UVec n a) = Vector_UVec {getUVecVector :: U.Vector a}

newtype instance U.MVector s (UVec n a) = MVector_UVec {getUVecMVector :: U.MVector s a}

instance (KnownNat n, U.Unbox a) => G.Vector U.Vector (UVec n a) where
  basicUnsafeFreeze = fmap Vector_UVec . G.basicUnsafeFreeze . getUVecMVector
  {-# INLINE basicUnsafeFreeze #-}
  basicUnsafeThaw = fmap MVector_UVec . G.basicUnsafeThaw . getUVecVector
  {-# INLINE basicUnsafeThaw #-}
  basicLength = (`quot` dimVal @n) . G.basicLength . getUVecVector
  {-# INLINE basicLength #-}
  basicUnsafeSlice = coerce (G.basicUnsafeSlice @U.Vector @a `on` (dimVal @n *))
  {-# INLINE basicUnsafeSlice #-}
  basicUnsafeIndexM (Vector_UVec raw) i =
    pure $
      Vec $
        M.fromUnboxedVector M.Par $
          U.unsafeSlice (i * dimVal @n) (dimVal @n) raw
  {-# INLINE basicUnsafeIndexM #-}

instance (KnownNat n, U.Unbox a) => MG.MVector U.MVector (UVec n a) where
  basicLength = (`quot` dimVal @n) . MG.basicLength . getUVecMVector
  {-# INLINE basicLength #-}
  basicUnsafeSlice = coerce (MG.basicUnsafeSlice @U.MVector @a `on` (dimVal @n *))
  {-# INLINE basicUnsafeSlice #-}
  basicOverlaps = coerce $ MG.overlaps @U.MVector @a
  {-# INLINE basicOverlaps #-}
  basicUnsafeNew = fmap MVector_UVec . MG.basicUnsafeNew . (dimVal @n *)
  {-# INLINE basicUnsafeNew #-}
  basicInitialize = MG.basicInitialize . getUVecMVector
  {-# INLINE basicInitialize #-}
  basicUnsafeRead (MVector_UVec raw) i =
    Vec . M.fromUnboxedVector M.Par
      <$> G.freeze
        (MG.unsafeSlice (dimVal @n * i) (dimVal @n) raw)
  {-# INLINE basicUnsafeRead #-}
  basicUnsafeWrite (MVector_UVec raw) i (Vec vec) =
    G.unsafeCopy (MG.unsafeSlice (dimVal @n * i) (dimVal @n) raw) $
      M.toUnboxedVector vec
  {-# INLINE basicUnsafeWrite #-}

instance (KnownNat n, U.Unbox a) => U.Unbox (UVec n a)

newtype instance U.Vector (UMat n m a) = Vector_UMat {getUMatVector :: U.Vector a}

newtype instance U.MVector s (UMat n m a) = MVector_UMat {getUMatMVector :: U.MVector s a}

instance (KnownNat n, KnownNat m, U.Unbox a) => G.Vector U.Vector (UMat n m a) where
  basicUnsafeFreeze = fmap Vector_UMat . G.basicUnsafeFreeze . getUMatMVector
  {-# INLINE basicUnsafeFreeze #-}
  basicUnsafeThaw = fmap MVector_UMat . G.basicUnsafeThaw . getUMatVector
  {-# INLINE basicUnsafeThaw #-}
  basicLength = (`quot` (dimVal @(m * n))) . G.basicLength . getUMatVector
  {-# INLINE basicLength #-}
  basicUnsafeSlice =
    coerce
      (G.basicUnsafeSlice @U.Vector @a `on` (dimVal @(m * n) *))
  {-# INLINE basicUnsafeSlice #-}
  basicUnsafeIndexM (Vector_UMat raw) i =
    pure $
      Mat $
        M.resize' (Sz2 (dimVal @m) (dimVal @n)) $
          M.fromUnboxedVector M.Par $
            U.unsafeSlice (i * dimVal @(m * n)) (dimVal @(m * n)) raw
  {-# INLINE basicUnsafeIndexM #-}

instance (KnownNat n, KnownNat m, U.Unbox a) => MG.MVector U.MVector (UMat n m a) where
  basicLength = (`quot` dimVal @(n * m)) . MG.basicLength . getUMatMVector
  {-# INLINE basicLength #-}
  basicUnsafeSlice = coerce (MG.basicUnsafeSlice @U.MVector @a `on` (dimVal @(m * n) *))
  {-# INLINE basicUnsafeSlice #-}
  basicOverlaps = coerce $ MG.overlaps @U.MVector @a
  {-# INLINE basicOverlaps #-}
  basicUnsafeNew = fmap MVector_UMat . MG.basicUnsafeNew . (dimVal @(m * n) *)
  {-# INLINE basicUnsafeNew #-}
  basicInitialize = MG.basicInitialize . getUMatMVector
  {-# INLINE basicInitialize #-}
  basicUnsafeRead (MVector_UMat raw) i =
    Mat . M.resize' (Sz2 (dimVal @m) (dimVal @n)) . M.fromUnboxedVector M.Par
      <$> G.freeze
        (MG.unsafeSlice (dimVal @(m * n) * i) (dimVal @(m * n)) raw)
  {-# INLINE basicUnsafeRead #-}
  basicUnsafeWrite (MVector_UMat raw) i (Mat vec) =
    G.unsafeCopy (MG.unsafeSlice (dimVal @(m * n) * i) (dimVal @(m * n)) raw) $
      M.toUnboxedVector vec
  {-# INLINE basicUnsafeWrite #-}

instance (KnownNat n, KnownNat m, U.Unbox a) => U.Unbox (UMat n m a)

{-
>>> fromBatchData  $ V.fromList [Linear.V2 0 1, Linear.V2 2 3, Linear.V2 3 (4 :: Double)]
MkSomeBatch (Mat {runMat = Array U Par (Sz (1 :. 6))
  [ [ 0.0, 1.0, 2.0, 3.0, 3.0, 4.0 ]
  ]})
-}

splitColAt ::
  forall n m l r a.
  (KnownNat n, M.Source r a) =>
  Mat r (n + m) l a ->
  (Mat M.D n l a, Mat M.D m l a)
{-# INLINE splitColAt #-}
splitColAt = coerce $ M.splitAt' 1 (dimVal @n)

splitRowAt ::
  forall n m l r a.
  (KnownNat n, M.Source r a) =>
  Mat r l (n + m) a ->
  (Mat M.D l n a, Mat M.D l m a)
{-# INLINE splitRowAt #-}
splitRowAt = coerce $ M.splitAt' 2 (dimVal @n)

{-
>>> splitColAt @1 $ Mat @M.U @4 @3 $ M.computeP $ 0 M...: (3 M.:. 4)
(Mat {runMat = Array D Seq (Sz (3 :. 1))
  [ [ 0 :. 0 ]
  , [ 1 :. 0 ]
  , [ 2 :. 0 ]
  ]},Mat {runMat = Array D Seq (Sz (3 :. 3))
  [ [ 0 :. 1, 0 :. 2, 0 :. 3 ]
  , [ 1 :. 1, 1 :. 2, 1 :. 3 ]
  , [ 2 :. 1, 2 :. 2, 2 :. 3 ]
  ]})

-}

fromRowMat ::
  forall v m t a.
  ( FromVec t
  , MU.Unbox a
  , M.Load (VM.ARepr v) M.Ix1 (t a)
  , M.Manifest (VM.ARepr v) (t a)
  , G.Vector v (t a)
  , v ~ VM.VRepr (VM.ARepr v)
  ) =>
  UMat m (Size t) a ->
  v (t a)
{-# INLINE fromRowMat #-}
fromRowMat =
  VM.toVector
    . M.computeP @(VM.ARepr v)
    . M.map (fromVec . Vec)
    . M.outerSlices
    . runMat

{-
>>> fromBatchData @Linear.V2 (V.fromList [1, 2, 3, 4])
MkSomeBatch (Mat {runMat = Array U Par (Sz (4 :. 2))
  [ [ 1.0, 1.0 ]
  , [ 2.0, 2.0 ]
  , [ 3.0, 3.0 ]
  , [ 4.0, 4.0 ]
  ]})
-}

deriving instance (Floating a, MU.Unbox a) => M.NumericFloat M.U a

-- | N.B. Checks header statically to prevent malformed deserialisation.
instance
  (KnownNat n, KnownNat m, Persist a, M.Manifest r a) =>
  Persist (Mat r n m a)
  where
  get = do
    let !expSize = Sz2 (dimVal @m) (dimVal @n)
    mat0 <- getArray
    when (M.size mat0 /= expSize) $
      fail $
        "Matrix size mismatched; expected: "
          <> show expSize
          <> ", but got: "
          <> show (M.size mat0)
    pure $ Mat mat0
  put = putArray . runMat
  {-# INLINE put #-}

-- | N.B. Checks header statically to prevent malformed deserialisation.
instance
  (KnownNat n, Persist a, M.Manifest r a) =>
  Persist (Vec r n a)
  where
  get = do
    let !expSize = Sz1 (dimVal @n)
    mat0 <- getArray
    when (M.size mat0 /= expSize) $
      fail $
        "Vector size mismatched; expected: "
          <> show expSize
          <> ", but got: "
          <> show (M.size mat0)
    pure $ Vec mat0
  {-# INLINE get #-}
  put = putArray . runVec
  {-# INLINE put #-}
