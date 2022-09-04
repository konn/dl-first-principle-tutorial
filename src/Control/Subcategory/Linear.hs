{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Control.Subcategory.Linear
  ( CAdditive (..),
    Vec (..),
    WrapLinear (..),
    Mat,
    (!.!),
    (!*!),
    (!*),
    V1,
    pattern V1,
    V2,
    pattern V2,
    V3,
    pattern V3,
    V4,
    pattern V4,
    UMat,
    HasSize (..),
    GHasSize (),
    FromVec (..),
    genericFromVec,
    GFromVec (..),
  )
where

import Control.Applicative (Applicative (..))
import Control.Lens (alaf, folded, iforMOf_)
import Control.Lens.Internal.Coerce
import Control.Monad (join)
import Control.Subcategory
import qualified Data.Bifunctor as Bi
import Data.Function (on)
import Data.Kind (Type)
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Core.Operations as M
import Data.Monoid (Sum (..))
import Data.Proxy (Proxy (..))
import Data.Sized (Sized, unsafeToSized', unsized)
import qualified Data.Sized as S
import Data.Strict (Pair (..), type (:!:))
import qualified Data.Strict as St
import Data.These (These (..))
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as MU
import GHC.Generics hiding (V1)
import GHC.TypeNats (KnownNat, Nat, natVal, type (*), type (+))
import Generic.Data
import qualified Linear
import qualified Linear.V as LinearV

infixl 6 ^+^, ^-^

infixl 7 *^, ^*, ^/, ^.^

class (HasSize t, CZip t, CRepeat t, CTraversable t) => CAdditive t where
  (^+^) :: (Dom t a, Num a) => t a -> t a -> t a
  {-# INLINE (^+^) #-}
  (^+^) = czipWith (+)
  (^-^) :: (Dom t a, Num a) => t a -> t a -> t a
  {-# INLINE (^-^) #-}
  (^-^) = czipWith (-)

  (*^) :: (Dom t a, Num a) => a -> t a -> t a
  {-# INLINE (*^) #-}
  (*^) = cmap . (*)

  (^*) :: (Dom t a, Num a) => t a -> a -> t a
  {-# INLINE (^*) #-}
  (^*) = flip $ cmap . flip (*)

  -- | Coordinate-wise multiplication
  (^.^) :: (Dom t a, Num a) => t a -> t a -> t a
  {-# INLINE (^.^) #-}
  (^.^) = czipWith (*)

  (^/) :: (Dom t a, Fractional a) => t a -> a -> t a
  {-# INLINE (^/) #-}
  (^/) = flip $ cmap . flip (/)

  zero :: (Dom t a, Num a) => t a
  {-# INLINE zero #-}
  zero = crepeat 0

  dim :: Proxy t -> Int
  default dim :: Proxy t -> Int
  {-# INLINE dim #-}
  dim = const $ fromIntegral $ natVal @(Size t) Proxy

  norm :: (Dom t a, Floating a) => t a -> a
  {-# INLINE norm #-}
  norm = sqrt . quadrance

  quadrance :: (Dom t a, Num a) => t a -> a
  {-# INLINE quadrance #-}
  quadrance = alaf Sum cfoldMap (join (*))

newtype WrapLinear (v :: Type -> Type) a = WrapLinear {runWrapLinear :: v a}
  deriving (Show, Eq, Ord, Generic, Generic1, Functor, Foldable, Traversable)
  deriving (Applicative) via Generically1 (WrapLinear v)
  deriving newtype (Linear.Additive)

deriving newtype instance (HasSize v) => HasSize (WrapLinear v)

instance Constrained (WrapLinear v) where
  type Dom (WrapLinear v) x = ()

instance Applicative v => CPointed (WrapLinear v) where
  cpure = pure
  {-# INLINE cpure #-}

instance Functor v => CFunctor (WrapLinear v) where
  cmap = fmap
  {-# INLINE cmap #-}

instance Applicative v => CSemialign (WrapLinear v) where
  calignWith f = liftA2 (fmap f . These)
  {-# INLINE calignWith #-}

instance Applicative v => CZip (WrapLinear v) where
  czipWith = liftA2
  {-# INLINE czipWith #-}

instance Applicative v => CRepeat (WrapLinear v) where
  crepeat = pure
  {-# INLINE crepeat #-}

instance Foldable v => CFoldable (WrapLinear v) where
  cfoldMap = foldMap
  {-# INLINE cfoldMap #-}

instance Traversable v => CTraversable (WrapLinear v) where
  ctraverse = traverse
  {-# INLINE ctraverse #-}

instance
  (HasSize v, Applicative v, Linear.Additive v, Traversable v) =>
  CAdditive (WrapLinear v)
  where
  (^+^) = (Linear.^+^)
  {-# INLINE (^+^) #-}
  (^-^) = (Linear.^-^)
  {-# INLINE (^-^) #-}
  (^.^) = Linear.liftI2 (*)
  {-# INLINE (^.^) #-}
  (^*) = (Linear.^*)
  {-# INLINE (^*) #-}
  (*^) = (Linear.*^)
  {-# INLINE (*^) #-}
  (^/) = (Linear.^/)
  {-# INLINE (^/) #-}
  dim = const $ fromIntegral $ natVal @(Size v) Proxy
  {-# INLINE dim #-}

instance KnownNat n => HasSize (Sized U.Vector n) where
  type Size (Sized U.Vector n) = n
  toVec = Vec
  {-# INLINE toVec #-}

instance (KnownNat n) => CAdditive (Sized U.Vector n) where
  dim = const $ fromIntegral $ natVal @n Proxy
  {-# INLINE dim #-}

type V1 = Vec 1

{-# COMPLETE V1 #-}

pattern V1 :: U.Unbox a => a -> V1 a
pattern V1 a = (Vec (a S.:< S.Nil))

{-# COMPLETE V2 #-}

type V2 = Vec 2

pattern V2 :: U.Unbox a => a -> a -> V2 a
pattern V2 a b = Vec (a S.:< b S.:< S.Nil)

type V3 = Vec 3

pattern V3 :: U.Unbox a => a -> a -> a -> V3 a
pattern V3 a b c = Vec (a S.:< b S.:< c S.:< S.Nil)

type V4 = Vec 4

pattern V4 :: U.Unbox a => a -> a -> a -> a -> V4 a
pattern V4 a b c d = Vec (a S.:< b S.:< c S.:< d S.:< S.Nil)

newtype Vec n a = Vec {getVec :: Sized U.Vector n a}
  deriving newtype (CFunctor, CSemialign, CPointed, CZip, CRepeat, CFoldable)

instance CTraversable (Vec n) where
  ctraverse f (Vec a) = Vec <$> ctraverse f a
  {-# INLINE ctraverse #-}

instance Constrained (Vec n) where
  type Dom (Vec n) a = U.Unbox a

instance KnownNat n => HasSize (Vec n) where
  type Size (Vec n) = n
  toVec = id
  {-# INLINE toVec #-}

type UMat = Mat M.U

newtype Mat r (n :: Nat) (m :: Nat) a = Mat {runMat :: M.Array r M.Ix2 a}
  deriving (Generic, Generic1)

deriving instance Show (M.Array r M.Ix2 a) => Show (Mat r n m a)

instance Constrained (Mat r n m) where
  type Dom (Mat r n m) a = (M.Load r M.Ix2 a, M.FoldNumeric r a, M.NumericFloat r a, M.Manifest r a)

instance CFunctor (Mat r n m) where
  cmap f (Mat arr) = Mat $ M.computeP $ M.map f arr
  {-# INLINE cmap #-}

instance (KnownNat n, KnownNat m) => CPointed (Mat r n m) where
  cpure = Mat . M.replicate M.Par (M.Sz2 (fromIntegral $ natVal $ Proxy @m) (fromIntegral $ natVal $ Proxy @n))
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
  clength = const $ fromIntegral $ natVal @n Proxy * natVal @m Proxy
  {-# INLINE clength #-}

instance (KnownNat n, KnownNat m) => CTraversable (Mat r n m) where
  ctraverse f = fmap Mat . M.traverseA f . runMat
  {-# INLINE ctraverse #-}

instance (KnownNat n, KnownNat m) => HasSize (UMat n m) where
  type Size (UMat n m) = n * m
  toVec = Vec . unsafeToSized' . M.toUnboxedVector . runMat
  {-# INLINE toVec #-}

instance (KnownNat n, KnownNat m) => CAdditive (UMat n m) where
  (^+^) = fmap Mat . ((M.!+!) `on` runMat)
  {-# INLINE (^+^) #-}
  (^-^) = fmap Mat . ((M.!+!) `on` runMat)
  {-# INLINE (^-^) #-}
  (*^) =
    coerce $ (M.*.) @M.Ix2 @M.U @a ::
      forall a. Dom (UMat n m) a => a -> UMat n m a -> UMat n m a
  {-# INLINE (*^) #-}
  (^*) =
    coerce $ (M..*) @M.Ix2 @M.U @a ::
      forall a. Dom (UMat n m) a => UMat n m a -> a -> UMat n m a
  {-# INLINE (^*) #-}
  (^/) =
    coerce $ (M../) @M.Ix2 @M.U @a ::
      forall a. Dom (UMat n m) a => UMat n m a -> a -> UMat n m a
  {-# INLINE (^/) #-}
  (^.^) = (!.!)
  {-# INLINE (^.^) #-}
  dim = const $ fromIntegral (natVal @n Proxy) * fromIntegral (natVal @m Proxy)
  {-# INLINE dim #-}

  norm = coerce M.normL2
  {-# INLINE norm #-}
  quadrance = coerce $ flip M.powerSumArray 2
  {-# INLINE quadrance #-}

-- | Matrix multiplication
(!*!) ::
  forall r n l m a.
  (M.Manifest r a, M.Numeric r a) =>
  Mat r n l a ->
  Mat r l m a ->
  Mat r n m a
(!*!) = coerce $ (M.!><!) @r @a

-- | Matrix-vector multiplication
(!*) ::
  forall n l a.
  (KnownNat l, U.Unbox a, Num a) =>
  Mat M.U n l a ->
  Vec n a ->
  Vec l a
Mat m !* v =
  let v' :: M.Vector M.U a
      !v' = M.fromUnboxedVector M.Par $ unsized $ getVec v
   in Vec $
        unsafeToSized' $
          M.toUnboxedVector $
            M.computeP $ m M.!>< v'

-- | Pointwise matrix-matrix mutlipliation
(!.!) ::
  forall r n m a.
  (M.Numeric r a) =>
  Mat r n m a ->
  Mat r n m a ->
  Mat r n m a
(!.!) = coerce $ (M.!*!) @M.Ix2 @r @a

class (KnownNat (Size f)) => HasSize (f :: Type -> Type) where
  type Size f :: Nat
  type Size f = GSize (Rep1 f)
  toVec :: U.Unbox a => f a -> Vec (Size f) a
  default toVec :: (Foldable f, U.Unbox a) => f a -> Vec (Size f) a
  {-# INLINE toVec #-}
  toVec xs = Vec $
    unsafeToSized' $
      U.create $ do
        mv <- MU.new $ fromIntegral $ natVal @(Size f) Proxy
        iforMOf_ folded xs $ MU.write mv
        pure mv

class HasSize f => FromVec f where
  fromVec :: U.Unbox a => Vec (Size f) a -> f a
  default fromVec ::
    (Generic1 f, GFromVec (Rep1 f)) =>
    U.Unbox a =>
    Vec (Size f) a ->
    f a
  {-# INLINE fromVec #-}
  fromVec = genericFromVec

genericFromVec :: (Generic1 f, GFromVec (Rep1 f)) => U.Unbox a => Vec (Size f) a -> f a
{-# INLINE genericFromVec #-}
genericFromVec = to1 . St.fst . gDecodeFrom . unsized . getVec

class (KnownNat (GSize f)) => GHasSize f where
  type GSize f :: Nat

class GHasSize f => GFromVec f where
  gDecodeFrom :: U.Unbox a => U.Vector a -> f a :!: U.Vector a

instance GHasSize f => GHasSize (M1 i c f) where
  type GSize (M1 i c f) = GSize f

instance GFromVec f => GFromVec (M1 i c f) where
  gDecodeFrom = coerce $ gDecodeFrom @f
  {-# INLINE gDecodeFrom #-}

instance (GHasSize f, GHasSize g) => GHasSize (f :*: g) where
  type GSize (f :*: g) = GSize f + GSize g

instance (GFromVec f, GFromVec g) => GFromVec (f :*: g) where
  gDecodeFrom xs =
    let (fa :!: rest) = gDecodeFrom xs
        (ga :!: rest') = gDecodeFrom rest
     in (fa :*: ga) :!: rest'
  {-# INLINE gDecodeFrom #-}

instance GHasSize Par1 where
  type GSize Par1 = 1

instance GFromVec Par1 where
  gDecodeFrom =
    coerce $ Bi.first U.head . St.toStrict . U.splitAt @a 1 ::
      forall a. U.Unbox a => U.Vector a -> Par1 a :!: U.Vector a
  {-# INLINE gDecodeFrom #-}

instance GHasSize (K1 i c) where
  type GSize (K1 i c) = 0

instance GHasSize U1 where
  type GSize U1 = 0

instance GFromVec U1 where
  gDecodeFrom = (U1 :!:)
  {-# INLINE gDecodeFrom #-}

deriving anyclass instance HasSize Linear.V0

deriving anyclass instance FromVec Linear.V0

deriving anyclass instance HasSize Linear.V1

deriving anyclass instance HasSize Linear.V2

deriving anyclass instance HasSize Linear.V3

deriving anyclass instance HasSize Linear.V4

instance KnownNat n => HasSize (LinearV.V n) where
  type Size (LinearV.V n) = n
  toVec = Vec . unsafeToSized' . U.convert . LinearV.toVector
  {-# INLINE toVec #-}
