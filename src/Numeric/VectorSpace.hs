{-# HLINT ignore "Redundant lambda" #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

module Numeric.VectorSpace
  ( VectorSpace (..),
    GVectorSpace (),
    GenericVectorSpace (),

    -- ** Deriving modifiers
    Scalar (..),
    WrapLinear (..),

    -- ** Generic operations
    genericReps,
    genericScaleL,
    genericScaleR,
    genericAddL,
    genericAddR,
    genericSubtL,
    genericSubtR,
    genericDivR,
    genericDot,
    genericSumS,
  )
where

import Data.Coerce (coerce)
import Data.Complex (Complex)
import Data.Function (on)
import Data.Functor.Const (Const (..))
import Data.Functor.Product (Product (..))
import Data.Kind
import Data.Monoid (Sum (..))
import GHC.Generics hiding (V1)
import Linear
import Numeric.Backprop (BVar, Backprop, Reifies, W, liftOp1, liftOp2, op1, op2)

infixl 6 .+, +., .-, -.

infixr 7 .*

infixl 7 /., *.

infixl 7 >.<

class Fractional k => VectorSpace k v | v -> k where
  reps :: k -> v
  default reps :: (Generic v, GVectorSpace k (Rep v)) => k -> v
  {-# INLINE reps #-}
  reps = genericReps

  (.*) :: k -> v -> v
  default (.*) :: (Generic v, GVectorSpace k (Rep v)) => k -> v -> v
  {-# INLINE (.*) #-}
  (.*) = genericScaleL

  (*.) :: v -> k -> v
  default (*.) :: (Generic v, GVectorSpace k (Rep v)) => v -> k -> v
  {-# INLINE (*.) #-}
  (*.) = genericScaleR

  (.+) :: k -> v -> v
  default (.+) :: (Generic v, GVectorSpace k (Rep v)) => k -> v -> v
  {-# INLINE (.+) #-}
  (.+) = genericAddL

  (+.) :: v -> k -> v
  default (+.) :: (Generic v, GVectorSpace k (Rep v)) => v -> k -> v
  {-# INLINE (+.) #-}
  (+.) = genericAddR

  (.-) :: k -> v -> v
  default (.-) :: (Generic v, GVectorSpace k (Rep v)) => k -> v -> v
  {-# INLINE (.-) #-}
  (.-) = genericSubtL

  (-.) :: v -> k -> v
  default (-.) :: (Generic v, GVectorSpace k (Rep v)) => v -> k -> v
  {-# INLINE (-.) #-}
  (-.) = genericSubtR

  (/.) :: v -> k -> v
  default (/.) :: (Generic v, GVectorSpace k (Rep v)) => v -> k -> v
  {-# INLINE (/.) #-}
  (/.) = genericDivR

  -- | Dot-product
  (>.<) :: v -> v -> k
  default (>.<) :: (Generic v, GVectorSpace k (Rep v)) => v -> v -> k
  {-# INLINE (>.<) #-}
  (>.<) = genericDot

  sumS :: v -> k
  default sumS :: (Generic v, GVectorSpace k (Rep v)) => v -> k
  {-# INLINE sumS #-}
  sumS = genericSumS

deriving anyclass instance
  (VectorSpace k a, VectorSpace k b) =>
  VectorSpace k (a, b)

deriving anyclass instance
  (VectorSpace k a, VectorSpace k b, VectorSpace k c) =>
  VectorSpace k (a, b, c)

deriving anyclass instance
  (VectorSpace k a, VectorSpace k b, VectorSpace k c, VectorSpace k d) =>
  VectorSpace k (a, b, c, d)

deriving anyclass instance
  (VectorSpace k a, VectorSpace k b, VectorSpace k c, VectorSpace k d, VectorSpace k e) =>
  VectorSpace k (a, b, c, d, e)

genericReps ::
  (Generic v, GVectorSpace k (Rep v)) =>
  k ->
  v
genericReps = to . greps

genericScaleL ::
  (Generic v, GVectorSpace k (Rep v)) =>
  k ->
  v ->
  v
{-# INLINE genericScaleL #-}
genericScaleL = (to .) . (. from) . gscaleL

genericScaleR ::
  (Generic v, GVectorSpace k (Rep v)) =>
  v ->
  k ->
  v
{-# INLINE genericScaleR #-}
genericScaleR = fmap to . gscaleR . from

genericAddL ::
  (Generic v, GVectorSpace k (Rep v)) =>
  k ->
  v ->
  v
{-# INLINE genericAddL #-}
genericAddL = (to .) . (. from) . gaddL

genericAddR ::
  (Generic v, GVectorSpace k (Rep v)) =>
  v ->
  k ->
  v
{-# INLINE genericAddR #-}
genericAddR = fmap to . gaddR . from

genericSubtL ::
  (Generic v, GVectorSpace k (Rep v)) =>
  k ->
  v ->
  v
{-# INLINE genericSubtL #-}
genericSubtL = (to .) . (. from) . gsubtL

genericSubtR ::
  (Generic v, GVectorSpace k (Rep v)) =>
  v ->
  k ->
  v
{-# INLINE genericSubtR #-}
genericSubtR = fmap to . gsubtR . from

genericDivR ::
  (Generic v, GVectorSpace k (Rep v)) =>
  v ->
  k ->
  v
{-# INLINE genericDivR #-}
genericDivR = fmap to . gdivR . from

genericDot ::
  (Generic v, GVectorSpace k (Rep v)) =>
  v ->
  v ->
  k
{-# INLINE genericDot #-}
genericDot = gdot `on` from

genericSumS ::
  (Generic v, GVectorSpace k (Rep v)) => v -> k
genericSumS = gsums . from

type GVectorSpace :: Type -> (Type -> Type) -> Constraint
class Fractional k => GVectorSpace k f where
  greps :: k -> f k
  gscaleL :: k -> f k -> f k
  gscaleR :: f k -> k -> f k
  gaddL :: k -> f k -> f k
  gaddR :: f k -> k -> f k
  gsubtL :: k -> f k -> f k
  gsubtR :: f k -> k -> f k
  gdivR :: f k -> k -> f k
  gdot :: f k -> f k -> k
  gsums :: f k -> k

instance GVectorSpace k f => GVectorSpace k (M1 i c f) where
  greps = coerce $ greps @k @f
  {-# INLINE greps #-}
  gscaleL = coerce $ gscaleL @k @f
  {-# INLINE gscaleL #-}
  gscaleR = coerce $ gscaleR @k @f
  {-# INLINE gscaleR #-}
  gaddL = coerce $ gaddL @k @f
  {-# INLINE gaddL #-}
  gaddR = coerce $ gaddR @k @f
  {-# INLINE gaddR #-}
  gsubtL = coerce $ gsubtL @k @f
  {-# INLINE gsubtL #-}
  gsubtR = coerce $ gsubtR @k @f
  {-# INLINE gsubtR #-}
  gdivR = coerce $ gdivR @k @f
  {-# INLINE gdivR #-}
  gdot = coerce $ gdot @k @f
  {-# INLINE gdot #-}
  gsums = coerce $ gsums @k @f
  {-# INLINE gsums #-}

bimapP :: (f a -> f' b) -> (g a -> g' b) -> (f :*: g) a -> (f' :*: g') b
{-# INLINE bimapP #-}
bimapP f g = \case (fa :*: ga) -> f fa :*: g ga

instance (GVectorSpace k f, GVectorSpace k g) => GVectorSpace k (f :*: g) where
  greps = (:*:) <$> greps <*> greps
  {-# INLINE greps #-}
  gscaleL = bimapP <$> gscaleL <*> gscaleL
  {-# INLINE gscaleL #-}
  gscaleR = \case
    (f :*: g) -> (:*:) <$> gscaleR f <*> gscaleR g
  {-# INLINE gscaleR #-}
  gaddL = bimapP <$> gaddL <*> gaddL
  {-# INLINE gaddL #-}
  gaddR = \case
    (f :*: g) -> (:*:) <$> gaddR f <*> gaddR g
  {-# INLINE gaddR #-}
  gsubtL = bimapP <$> gsubtL <*> gsubtL
  {-# INLINE gsubtL #-}
  gsubtR = \case
    (f :*: g) -> (:*:) <$> gsubtR f <*> gsubtR g
  {-# INLINE gsubtR #-}
  gdivR = \case
    (f :*: g) -> (:*:) <$> gdivR f <*> gdivR g
  {-# INLINE gdivR #-}
  gdot = \(f :*: g) (f' :*: g') -> gdot f f' + gdot g g'
  {-# INLINE gdot #-}
  gsums =
    getSum . getConst . sequenceA
      . bimapP (Const . Sum . gsums) (Const . Sum . gsums)
  {-# INLINE gsums #-}

instance Fractional k => GVectorSpace k U1 where
  greps = pure U1
  {-# INLINE greps #-}
  gscaleL = mempty
  {-# INLINE gscaleL #-}
  gscaleR = mempty
  {-# INLINE gscaleR #-}
  gaddL = mempty
  {-# INLINE gaddL #-}
  gaddR = mempty
  {-# INLINE gaddR #-}
  gsubtL = mempty
  {-# INLINE gsubtL #-}
  gsubtR = mempty
  {-# INLINE gsubtR #-}
  gdivR = mempty
  {-# INLINE gdivR #-}
  gdot = const $ const 0
  {-# INLINE gdot #-}
  gsums = const 0
  {-# INLINE gsums #-}

instance (VectorSpace k c) => GVectorSpace k (K1 i c) where
  greps = coerce $ reps @k @c
  {-# INLINE greps #-}
  gscaleL = coerce $ (.*) @k @c
  {-# INLINE gscaleL #-}
  gscaleR = coerce $ (*.) @k @c
  {-# INLINE gscaleR #-}
  gaddL = coerce $ (.+) @k @c
  {-# INLINE gaddL #-}
  gaddR = coerce $ (+.) @k @c
  {-# INLINE gaddR #-}
  gsubtL = coerce $ (.-) @k @c
  {-# INLINE gsubtL #-}
  gsubtR = coerce $ (-.) @k @c
  {-# INLINE gsubtR #-}
  gdivR = coerce $ (/.) @k @c
  {-# INLINE gdivR #-}
  gdot = coerce $ (>.<) @k @c
  {-# INLINE gdot #-}
  gsums = coerce $ sumS @k @c
  {-# INLINE gsums #-}

type GenericVectorSpace k a = (Generic a, GVectorSpace k (Rep a))

newtype Scalar a = Scalar {runScalar :: a}
  deriving (Show, Eq, Ord, Generic)
  deriving newtype
    ( Num
    , Integral
    , Real
    , RealFloat
    , Fractional
    , Floating
    , RealFrac
    , Enum
    , Bounded
    )

instance Fractional k => VectorSpace k (Scalar k) where
  reps = Scalar
  {-# INLINE reps #-}
  (.*) = coerce $ (*) @k
  {-# INLINE (.*) #-}
  (*.) = coerce $ (*) @k
  {-# INLINE (*.) #-}
  (.+) = coerce $ (+) @k
  {-# INLINE (.+) #-}
  (+.) = coerce $ (+) @k
  {-# INLINE (+.) #-}
  (.-) = coerce $ (-) @k
  {-# INLINE (.-) #-}
  (-.) = coerce $ (-) @k
  {-# INLINE (-.) #-}
  (/.) = coerce $ (/) @k
  {-# INLINE (/.) #-}
  (>.<) = coerce $ (*) @k
  {-# INLINE (>.<) #-}
  sumS = runScalar
  {-# INLINE sumS #-}

newtype WrapLinear v k = WrapLinear {runWrapLinear :: v k}
  deriving (Show, Eq, Ord, Generic, Traversable)
  deriving newtype
    ( Num
    , Integral
    , Real
    , RealFloat
    , Fractional
    , Floating
    , RealFrac
    , Enum
    , Bounded
    , Functor
    , Applicative
    , Foldable
    , Additive
    , Metric
    )

instance
  (Fractional k, Applicative v, Metric v, Foldable v) =>
  VectorSpace k (WrapLinear v k)
  where
  reps = pure
  {-# INLINE reps #-}
  (.*) = coerce $ (*^) @v @k
  {-# INLINE (.*) #-}
  (*.) = coerce $ (^*) @v @k
  {-# INLINE (*.) #-}
  (.+) = coerce $ fmap @v . (+) @k
  {-# INLINE (.+) #-}
  (+.) = coerce $ flip $ fmap @v . flip @k (+)
  {-# INLINE (+.) #-}
  (.-) = coerce $ fmap @v . (-) @k
  {-# INLINE (.-) #-}
  (-.) = coerce $ flip $ fmap @v . flip @k (-)
  {-# INLINE (-.) #-}
  (/.) = coerce $ (^/) @v @k
  {-# INLINE (/.) #-}
  (>.<) = coerce $ dot @v @k
  {-# INLINE (>.<) #-}
  sumS = coerce $ sum @v @k
  {-# INLINE sumS #-}

deriving anyclass instance Fractional k => VectorSpace k (V0 k)

deriving via WrapLinear V1 k instance Fractional k => VectorSpace k (V1 k)

deriving via WrapLinear V2 k instance Fractional k => VectorSpace k (V2 k)

deriving via WrapLinear V3 k instance Fractional k => VectorSpace k (V3 k)

deriving via WrapLinear V4 k instance Fractional k => VectorSpace k (V4 k)

deriving anyclass instance
  (VectorSpace k (f a), VectorSpace k (g a)) =>
  VectorSpace k (Product f g a)

instance
  (Reifies s W, Floating a, Backprop k, Backprop a, VectorSpace k a) =>
  VectorSpace (BVar s k) (BVar s a)
  where
  {-# INLINE reps #-}
  reps = liftOp1 $
    op1 $ \x -> (reps x, const 0)
  {-# INLINE (.*) #-}
  (.*) = liftOp2 $
    op2 $ \c v ->
      (c .* v, \dz -> (dz >.< v, c .* dz))
  {-# INLINE (*.) #-}
  (*.) = flip (.*)
  {-# INLINE (.+) #-}
  (.+) = liftOp2 $
    op2 $ \c v ->
      (c .+ v, \dz -> (sumS dz, dz))
  {-# INLINE (+.) #-}
  (+.) = flip (.+)
  {-# INLINE (.-) #-}
  (.-) = liftOp2 $
    op2 $ \c v ->
      (c .- v, \dz -> (sumS dz, -dz))
  {-# INLINE (-.) #-}
  (-.) = liftOp2 $
    op2 $ \v c ->
      (v -. c, \dz -> (dz, -sumS dz))
  {-# INLINE (/.) #-}
  (/.) = liftOp2 $
    op2 $ \v c ->
      (v /. c, \dz -> (dz /. c, (-dz >.< v) / (c * c)))
  {-# INLINE sumS #-}
  sumS = liftOp1 $ op1 $ \x -> (sumS x, reps)
  {-# INLINE (>.<) #-}
  (>.<) = liftOp2 $
    op2 $ \x y ->
      (x >.< y, \dz -> (dz .* y, dz .* x))

deriving via Scalar Double instance VectorSpace Double Double

deriving via Scalar Float instance VectorSpace Float Float

deriving via Scalar Rational instance VectorSpace Rational Rational

deriving via
  Scalar (Complex a)
  instance
    RealFloat a => VectorSpace (Complex a) (Complex a)
