{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module DeepLearning.NeuralNetowrk.Massiv
  ( -- * Central data-types
    NeuralNetwork,
    SomeNetwork (..),
    simpleSomeNetwork,
    activationVal,
    withSimpleNetwork,
    Network (..),
    Layer (..),
    Activation (..),
    SomeActivation (..),
    someActivation,
    type L,
    LayerKind (..),
    NetworkStat (..),
    networkStat,
    LayerInfo (..),

    -- ** Network construction
    LayerSpec (..),
    randomNetwork,
    affine,
    linear,
    sigmoid_,
    reLU_,
    tanh_,
    passthru_,
    batchnorm,

    -- * Training
    LossFunction,
    crossEntropy,
    trainGD,
    trainGDF,
    trainGD_,
    AdamParams (..),
    trainAdam,
    trainAdamF,
    trainAdam_,

    -- * Evaluation
    evalBatchNN,
    evalBatchF,
    evalNN,
    evalF,

    -- * General execution with backpropagation
    Pass (..),
    runNN,
    runLayer,
    gradNN,
    GradStack,
    toGradientStack,
    fromGradientStack,
    SLayerKind (..),
    KnownLayerKind (..),
    LayerLike,
    Aff,
    Lin,
    Act,
    BN,

    -- * Operators for manipulating networks
    withKnownNetwork,
    mapNetwork,
    htraverseNetwork,
    zipNetworkWith,
    zipNetworkWith3,
    KnownNetwork (..),
    NetworkShape (..),
    foldMapNetwork,
    foldZipNetwork,
  )
where

import Control.Lens (Lens', lens)
import Control.Subcategory (CZip (..), cmap)
import Control.Subcategory.Linear
import Data.Coerce (coerce)
import qualified Data.DList as DL
import Data.Functor.Product (Product)
import Data.Generics.Labels ()
import Data.Kind (Type)
import Data.List (iterate')
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array.Manifest.Vector as VM
import Data.Monoid (Sum (..))
import Data.Proxy (Proxy)
import Data.Strict (Pair (..))
import qualified Data.Strict.Tuple as SP
import Data.Type.Natural
import qualified Data.Vector.Generic as G
import Data.Vector.Orphans
import qualified Data.Vector.Unboxed as U
import Generic.Data
import Numeric.Backprop
import Numeric.Function.Activation (sigmoid)
import Numeric.Natural (Natural)
import System.Random.MWC.Distributions (normal, standard)
import System.Random.Stateful

data LayerKind = Aff | Lin | Act Activation | BN
  deriving (Show, Eq, Ord, Generic)

data SActivation (a :: Activation) where
  SReLU :: SActivation 'ReLU
  SSigmoid :: SActivation 'Sigmoid
  STanh :: SActivation 'Tanh
  SId :: SActivation 'Id

deriving instance Show (SActivation a)

class KnownActivation (a :: Activation) where
  sActivation :: SActivation a

instance KnownActivation 'ReLU where
  sActivation = SReLU

instance KnownActivation 'Sigmoid where
  sActivation = SSigmoid

instance KnownActivation 'Tanh where
  sActivation = STanh

instance KnownActivation 'Id where
  sActivation = SId

data SLayerKind (l :: LayerKind) n m where
  SAff :: SLayerKind 'Aff n m
  SLin :: SLayerKind 'Lin n m
  SAct :: SActivation a -> SLayerKind ( 'Act a) n n
  SBN :: SLayerKind 'BN n n

class
  (KnownNat n, KnownNat m) =>
  KnownLayerKind (l :: LayerKind) (n :: Nat) (m :: Nat)
  where
  sLayerKind :: SLayerKind l n m

instance
  (KnownNat n, KnownNat m) =>
  KnownLayerKind 'Aff n m
  where
  sLayerKind = SAff

instance
  (KnownNat n, KnownNat m) =>
  KnownLayerKind 'Lin n m
  where
  sLayerKind = SLin

instance (n ~ m, KnownNat m, KnownActivation a) => KnownLayerKind ( 'Act a) n m where
  sLayerKind = SAct sActivation

instance (n ~ m, KnownNat m) => KnownLayerKind 'BN n m where
  sLayerKind = SBN

type Aff = 'Aff

type Lin = 'Lin

type Act = 'Act

type BN = 'BN

type LayerLike = LayerKind -> Nat -> Nat -> Type -> Type

data Activation = ReLU | Sigmoid | Tanh | Id
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

data SomeActivation where
  MkSomeActivation :: KnownActivation act => SActivation act -> SomeActivation

deriving instance Show SomeActivation

activationVal :: SActivation act -> Activation
activationVal SReLU = ReLU
activationVal SSigmoid = Sigmoid
activationVal STanh = Tanh
activationVal SId = Id

someActivation :: Activation -> SomeActivation
someActivation ReLU = MkSomeActivation SReLU
someActivation Sigmoid = MkSomeActivation SSigmoid
someActivation Tanh = MkSomeActivation STanh
someActivation Id = MkSomeActivation SId

type LayerSpec :: LayerLike
data LayerSpec l n m a where
  AffP :: a -> LayerSpec 'Aff n m a
  LinP :: a -> LayerSpec 'Lin n m a
  ActP :: KnownActivation act => LayerSpec ( 'Act act) n n a
  BNP :: a -> LayerSpec 'BN n n a

deriving instance Show a => Show (LayerSpec l n m a)

deriving instance Eq a => Eq (LayerSpec l n m a)

deriving instance Ord a => Ord (LayerSpec l n m a)

affine :: forall o i a. a -> LayerSpec 'Aff i o a
affine = AffP

linear :: forall o i a. a -> LayerSpec 'Lin i o a
linear = LinP

sigmoid_ :: forall i a. LayerSpec (Act 'Sigmoid) i i a
sigmoid_ = ActP

reLU_ :: forall i a. LayerSpec (Act 'ReLU) i i a
reLU_ = ActP

tanh_ :: forall i a. LayerSpec (Act 'Tanh) i i a
tanh_ = ActP

passthru_ :: forall i a. LayerSpec (Act 'Id) i i a
passthru_ = ActP

batchnorm :: forall i a. a -> LayerSpec 'BN i i a
batchnorm = BNP

type Activation' :: (Type -> Type) -> LayerKind -> Nat -> Nat -> Type -> Type
data Activation' m l i o a = Activation' {getActivation :: Activation, seed :: m a}
  deriving (Show, Eq, Ord, Generic, Bounded, Functor, Foldable, Traversable)

type Layer :: LayerKind -> Nat -> Nat -> Type -> Type
data Layer l i o a where
  Affine :: !(UMat i o a) -> !(UVec o a) -> Layer Aff i o a
  Linear :: !(UMat i o a) -> Layer Lin i o a
  Activate :: !(SActivation act) -> Layer (Act act) i i a
  BatchNorm :: !(UVec i a) -> !(UVec i a) -> !(UVec i a) -> !(UVec i a) -> Layer BN i i a

instance
  (KnownLayerKind l i o, KnownNat i, KnownNat o, Num a, U.Unbox a) =>
  Num (Layer l i o a)
  where
  {-# INLINE fromInteger #-}
  fromInteger = case sLayerKind @l @i @o of
    SAff -> Affine <$> fromInteger <*> fromInteger
    SLin -> Linear <$> fromInteger
    SAct sact -> const $ Activate sact
    SBN -> BatchNorm <$> fromInteger <*> fromInteger <*> fromInteger <*> fromInteger
  (+) = liftBinW (+) (+)
  {-# INLINE (+) #-}
  (-) = liftBinW (-) (-)
  {-# INLINE (-) #-}
  (*) = liftBinW (*) (*)
  {-# INLINE (*) #-}
  negate = liftUnW negate negate
  {-# INLINE negate #-}
  signum = liftUnW signum signum
  {-# INLINE signum #-}
  abs = liftUnW abs abs
  {-# INLINE abs #-}

instance
  (KnownLayerKind l i o, KnownNat i, KnownNat o, Floating a, U.Unbox a) =>
  Fractional (Layer l i o a)
  where
  fromRational = case sLayerKind @l @i @o of
    SAff -> Affine <$> fromRational <*> fromRational
    SLin -> Linear <$> fromRational
    SAct sact -> const $ Activate sact
    SBN -> BatchNorm <$> fromRational <*> fromRational <*> fromRational <*> fromRational
  {-# INLINE fromRational #-}
  recip = liftUnW recip recip
  {-# INLINE recip #-}
  (/) = liftBinW (/) (/)
  {-# INLINE (/) #-}

instance
  (KnownLayerKind l i o, KnownNat i, KnownNat o, Floating a, U.Unbox a) =>
  Floating (Layer l i o a)
  where
  pi = case sLayerKind @l @i @o of
    SAff -> Affine pi pi
    SLin -> Linear pi
    SAct sact -> Activate sact
    SBN -> BatchNorm pi pi pi pi
  {-# INLINE pi #-}
  exp = liftUnW exp exp
  {-# INLINE exp #-}
  log = liftUnW log log
  {-# INLINE log #-}
  sin = liftUnW sin sin
  {-# INLINE sin #-}
  cos = liftUnW cos cos
  {-# INLINE cos #-}
  tan = liftUnW tan tan
  {-# INLINE tan #-}
  asin = liftUnW asin asin
  {-# INLINE asin #-}
  acos = liftUnW acos acos
  {-# INLINE acos #-}
  atan = liftUnW atan atan
  {-# INLINE atan #-}
  sinh = liftUnW sinh sinh
  {-# INLINE sinh #-}
  cosh = liftUnW cosh cosh
  {-# INLINE cosh #-}
  tanh = liftUnW tanh tanh
  {-# INLINE tanh #-}
  asinh = liftUnW asinh asinh
  {-# INLINE asinh #-}
  acosh = liftUnW acosh acosh
  {-# INLINE acosh #-}
  atanh = liftUnW atanh atanh
  {-# INLINE atanh #-}

newtype Grads l i o a = Grads (Layer l i o a)
  deriving newtype (Show)

deriving newtype instance
  (KnownNat i, KnownNat o, U.Unbox a, Num a) => Backprop (Grads l i o a)

deriving newtype instance
  (KnownNat i, KnownNat o, U.Unbox a, Floating a, KnownLayerKind l i o) =>
  VectorSpace a (Grads l i o a)

type GradStack = Network Grads

liftBinW ::
  (UVec o a -> UVec o a -> UVec o a) ->
  (UMat i o a -> UMat i o a -> UMat i o a) ->
  Layer l i o a ->
  Layer l i o a ->
  Layer l i o a
{-# INLINE liftBinW #-}
liftBinW fV fM (Affine mat vec) (Affine mat' vec') =
  Affine (fM mat mat') (fV vec vec')
liftBinW _ fM (Linear mat) (Linear mat') = Linear $ fM mat mat'
liftBinW _ _ l@Activate {} Activate {} = l
liftBinW fV _ (BatchNorm mu sigma w b) (BatchNorm mu' sigma' w' b') =
  BatchNorm (fV mu mu') (fV sigma sigma') (fV w w') (fV b b')

liftUnW ::
  (UVec o a -> UVec o a) ->
  (UMat i o a -> UMat i o a) ->
  Layer l i o a ->
  Layer l i o a
{-# INLINE liftUnW #-}
liftUnW fV fM = \case
  (Affine mat vec) -> Affine (fM mat) (fV vec)
  (Linear mat) -> Linear (fM mat)
  l@Activate {} -> l
  (BatchNorm vec vec' vec2 vec3) ->
    BatchNorm (fV vec) (fV vec') (fV vec2) (fV vec3)

instance
  (Floating a, U.Unbox a, KnownNat i, KnownNat o, KnownLayerKind l i o) =>
  VectorSpace a (Layer l i o a)
  where
  reps x = case sLayerKind @l @i @o of
    SAff -> Affine (reps x) (reps x)
    SLin -> Linear (reps x)
    SAct sact -> Activate sact
    SBN -> BatchNorm (reps x) (reps x) (reps x) (reps x)
  (.*) = liftUnW <$> (.*) <*> (.*)
  {-# INLINE (.*) #-}
  (.+) = liftUnW <$> (.+) <*> (.+)
  {-# INLINE (.+) #-}
  (+.) = flip $ liftUnW <$> flip (+.) <*> flip (+.)
  {-# INLINE (+.) #-}
  (.-) = liftUnW <$> (.-) <*> (.-)
  {-# INLINE (.-) #-}
  (-.) = flip $ liftUnW <$> flip (-.) <*> flip (-.)
  {-# INLINE (-.) #-}
  (*.) = flip $ liftUnW <$> flip (*.) <*> flip (*.)
  {-# INLINE (*.) #-}
  (/.) = flip $ liftUnW <$> flip (/.) <*> flip (/.)
  {-# INLINE (/.) #-}
  (Affine mat vec) >.< (Affine mat' vec') =
    mat >.< mat' + vec >.< vec'
  (Linear mat) >.< (Linear mat') = mat >.< mat'
  Activate {} >.< Activate {} = 0
  (BatchNorm m s w b) >.< (BatchNorm m' s' w' b') = m >.< m' + s >.< s' + w >.< w' + b >.< b'
  {-# INLINE (>.<) #-}
  sumS (Affine mat vec) = sumS mat + sumS vec
  sumS (Linear mat) = sumS mat
  sumS Activate {} = 0
  sumS (BatchNorm m s w b) = sumS m + sumS s + sumS w + sumS b
  {-# INLINE sumS #-}

deriving instance (U.Unbox a, Show a) => Show (Layer l i o a)

data Pass = Train | Eval
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

affineMatL :: Lens' (Layer Aff i o a) (UMat i o a)
{-# INLINE affineMatL #-}
affineMatL = lens (\case (Affine mat _) -> mat) $
  \case (Affine _ vec) -> (`Affine` vec)

affineBiasL :: Lens' (Layer Aff i o a) (UVec o a)
{-# INLINE affineBiasL #-}
affineBiasL = lens (\case (Affine _ v) -> v) $
  \case (Affine mat _) -> Affine mat

linearMatL :: Lens' (Layer Lin i o a) (UMat i o a)
{-# INLINE linearMatL #-}
linearMatL = lens (\case (Linear mat) -> mat) $ const Linear

data BatchLayer i a = BatchLayer {mean, deviation, scale, shift :: UVec i a}
  deriving (Show, Generic)

deriving anyclass instance
  (KnownNat i, M.Numeric M.U a, U.Unbox a) => Backprop (BatchLayer i a)

batchNormL :: Lens' (Layer BN i i a) (BatchLayer i a)
batchNormL =
  lens
    ( \(BatchNorm mean deviation scale shift) ->
        BatchLayer {..}
    )
    $ const $ \BatchLayer {..} ->
      BatchNorm mean deviation scale shift

runLayer ::
  forall s m l a i o.
  ( U.Unbox a
  , KnownLayerKind l i o
  , KnownNat m -- batch size
  , RealFloat a
  , Backprop a
  , Reifies s W
  ) =>
  Pass ->
  BVar s (Layer l i o a) ->
  BVar s (UMat m i a) ->
  BVar s (UMat m o a)
{-# INLINE runLayer #-}
runLayer pass = case sLayerKind @l @i @o of
  SAff -> \aff x ->
    let w = aff ^^. affineMatL
        b = aff ^^. affineBiasL
     in w !*!: x + duplicateAsCols' b
  SLin -> \lin x ->
    let w = lin ^^. linearMatL
     in w !*!: x
  SAct a -> const $ applyActivatorBV a
  SBN -> \lay ->
    let !bnParams = lay ^^. batchNormL
        !mu = bnParams ^^. #mean
        !sigma = bnParams ^^. #deviation
        !gamma = bnParams ^^. #scale
        !beta = bnParams ^^. #shift
     in case pass of
          Train -> \x ->
            let !m = fromIntegral $ dimVal @m
                batchMu = sumRows' x /. m
                xRel = x - duplicateAsCols' batchMu
                batchSigma = sumRows' (xRel * xRel) /. m
                ivar = recip $ sqrt $ batchSigma +. 1e-12
                gammax =
                  duplicateAsCols' gamma * xRel * duplicateAsCols' ivar
             in gammax + duplicateAsCols' beta
          Eval -> \x ->
            let eps = 1e-12
                out1 =
                  (x - duplicateAsCols' mu)
                    / duplicateAsCols' (sqrt (sigma +. eps))
             in duplicateAsCols' gamma * out1 + duplicateAsCols' beta

applyActivatorBV ::
  ( Reifies s W
  , KnownNat n
  , KnownNat m
  , M.Load r M.Ix2 a
  , M.NumericFloat r a
  , RealFloat a
  , M.Manifest r a
  ) =>
  SActivation act ->
  BVar s (Mat r n m a) ->
  BVar s (Mat r n m a)
applyActivatorBV SReLU = liftOp1 $
  op1 $ \mat ->
    ( cmap (max 0) mat
    , czipWith
        ( \x d ->
            if x < 0 then 0 else d
        )
        mat
    )
applyActivatorBV SSigmoid = sigmoid
applyActivatorBV STanh = tanh
applyActivatorBV SId = id

-- | FIXME: defining on the case of @l@ would reduce runtime branching.
instance (KnownNat i, KnownNat o, U.Unbox a, Num a) => Backprop (Layer l i o a) where
  zero = \case
    Affine {} -> Affine 0 0
    Linear {} -> Linear 0
    l@Activate {} -> l
    BatchNorm {} -> BatchNorm 0 0 0 0
  {-# INLINE zero #-}
  one = \case
    Affine {} -> Affine 1 1
    Linear {} -> Linear 1
    l@Activate {} -> l
    BatchNorm {} -> BatchNorm 1 1 1 1
  {-# INLINE one #-}
  add (Affine mat vec) (Affine mat' vec') = Affine (add mat mat') (add vec vec')
  add (Linear mat) (Linear mat') = Linear (add mat mat')
  add l@Activate {} Activate {} = l
  add (BatchNorm a b c d) (BatchNorm a' b' c' d') =
    BatchNorm (a + a') (b + b') (c + c') (d + d')
  {-# INLINE add #-}

data Spec = L LayerKind Nat
  deriving (Generic)

type L = 'L

type Network ::
  LayerLike ->
  Nat ->
  [Spec] ->
  Nat ->
  Type ->
  Type
data Network h i fs o a where
  Output :: Network h i '[] i a
  (:-) ::
    (KnownNat k, KnownLayerKind l i k) =>
    !(h l i k a) ->
    !(Network h k fs o a) ->
    Network h i (L l k ': fs) o a

infixr 9 :-

type NeuralNetwork = Network Layer

instance
  (KnownNat i, KnownNat o, U.Unbox a, Num a, KnownNetwork i fs o) =>
  Num (Network Layer i fs o a)
  where
  fromInteger = go $ networkShape @i @fs @o
    where
      go :: KnownNat l => NetworkShape l hs o -> Integer -> Network Layer l hs o a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons rest) = (:-) <$> fromInteger <*> go rest
  (+) = zipNetworkWith (+)
  {-# INLINE (+) #-}
  (-) = zipNetworkWith (-)
  {-# INLINE (-) #-}
  (*) = zipNetworkWith (*)
  {-# INLINE (*) #-}
  negate = mapNetwork negate
  {-# INLINE negate #-}
  abs = mapNetwork abs
  {-# INLINE abs #-}
  signum = mapNetwork signum
  {-# INLINE signum #-}

instance
  ( KnownNat i
  , KnownNat o
  , U.Unbox a
  , Num a
  , KnownNetwork i fs o
  , Floating a
  ) =>
  VectorSpace a (Network Layer i fs o a)
  where
  reps = go $ networkShape @i @fs @o
    where
      go :: KnownNat l => NetworkShape l hs o -> a -> Network Layer l hs o a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons rest) = (:-) <$> reps <*> go rest
  {-# INLINE reps #-}
  (.*) c = mapNetwork (c .*)
  {-# INLINE (.*) #-}
  (+.) = flip $ \c -> mapNetwork (+. c)
  {-# INLINE (+.) #-}
  (.+) c = mapNetwork (c .+)
  {-# INLINE (.+) #-}
  (-.) = flip $ \c -> mapNetwork (-. c)
  {-# INLINE (-.) #-}
  (.-) c = mapNetwork (c .-)
  {-# INLINE (.-) #-}
  (*.) = flip $ \c -> mapNetwork (*. c)
  {-# INLINE (*.) #-}
  (/.) = flip $ \c -> mapNetwork (/. c)
  {-# INLINE (/.) #-}
  (>.<) = fmap getSum . foldZipNetwork (\l r -> Sum $ l >.< r)
  {-# INLINE (>.<) #-}
  sumS = getSum . foldMapNetwork (Sum . sumS)
  {-# INLINE sumS #-}

instance
  ( KnownNat i
  , KnownNat o
  , U.Unbox a
  , Floating a
  , KnownNetwork i fs o
  ) =>
  Fractional (Network Layer i fs o a)
  where
  fromRational = go $ networkShape @i @fs @o
    where
      go :: KnownNat l => NetworkShape l hs o -> Rational -> Network Layer l hs o a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons rest) = (:-) <$> fromRational <*> go rest
  (/) = zipNetworkWith (/)
  {-# INLINE (/) #-}
  recip = mapNetwork recip
  {-# INLINE recip #-}

instance
  ( KnownNat i
  , KnownNat o
  , U.Unbox a
  , Floating a
  , KnownNetwork i fs o
  ) =>
  Floating (Network Layer i fs o a)
  where
  pi = go $ networkShape @i @fs @o
    where
      go :: KnownNat l => NetworkShape l hs o -> Network Layer l hs o a
      {-# INLINE go #-}
      go IsOutput = Output
      go (IsCons rest) = pi :- go rest
  {-# INLINE pi #-}
  exp = mapNetwork exp
  {-# INLINE exp #-}
  log = mapNetwork log
  {-# INLINE log #-}
  sin = mapNetwork sin
  {-# INLINE sin #-}
  cos = mapNetwork cos
  {-# INLINE cos #-}
  tan = mapNetwork tan
  {-# INLINE tan #-}
  asin = mapNetwork asin
  {-# INLINE asin #-}
  acos = mapNetwork acos
  {-# INLINE acos #-}
  atan = mapNetwork atan
  {-# INLINE atan #-}
  sinh = mapNetwork sinh
  {-# INLINE sinh #-}
  cosh = mapNetwork cosh
  {-# INLINE cosh #-}
  tanh = mapNetwork tanh
  {-# INLINE tanh #-}
  asinh = mapNetwork asinh
  {-# INLINE asinh #-}
  acosh = mapNetwork acosh
  {-# INLINE acosh #-}
  atanh = mapNetwork atanh
  {-# INLINE atanh #-}

instance
  ( forall l f g. (KnownNat f, KnownNat g, U.Unbox a) => Show (h l f g a)
  , KnownNat i
  , KnownNat o
  , U.Unbox a
  ) =>
  Show (Network h i fs o a)
  where
  showsPrec _ Output = showString "Output"
  showsPrec d (hlika :- net') =
    showParen (d > 9) $
      showsPrec 10 hlika . showString " :- " . showsPrec 9 net'

data NetworkShape i xs o where
  IsOutput :: NetworkShape i '[] i
  IsCons :: (KnownLayerKind l i k, KnownNat k) => NetworkShape k hs o -> NetworkShape i (L l k ': hs) o

class (KnownNat i, KnownNat o) => KnownNetwork i (xs :: [Spec]) o where
  networkShape :: NetworkShape i xs o

instance (i ~ o, KnownNat o) => KnownNetwork i '[] o where
  networkShape = IsOutput

instance
  ( KnownNat i
  , KnownLayerKind l i k
  , KnownNetwork k xs o
  ) =>
  KnownNetwork i (L l k : xs) o
  where
  networkShape = IsCons networkShape

data TopLayerView h i l k xs o a = TopLayerView
  { topLayer :: !(h l i k a)
  , continue :: Network h k xs o a
  }
  deriving (Generic)

deriving anyclass instance
  ( U.Unbox a
  , Num a
  , KnownNat i
  , KnownNat k
  , KnownNat o
  ) =>
  Backprop (TopLayerView Layer i l k xs o a)

topLayerL ::
  (KnownLayerKind l i k) =>
  Lens' (Network h i (L l k ': xs) o a) (TopLayerView h i l k xs o a)
topLayerL =
  lens
    ( \(hlika :- net) ->
        TopLayerView {topLayer = hlika, continue = net}
    )
    (\_ TopLayerView {..} -> topLayer :- continue)

instance
  ( KnownNat i
  , KnownNat o
  , U.Unbox a
  , Num a
  ) =>
  Backprop (Network Layer i ls o a)
  where
  zero Output = Output
  zero (h :- hs) = zero h :- zero hs
  one Output = Output
  one (h :- hs) = one h :- one hs
  add Output Output = Output
  add (h :- hs) (g :- gs) = add h g :- add hs gs

runNN ::
  forall m i hs o a s.
  ( KnownNat m
  , U.Unbox a
  , RealFloat a
  , Backprop a
  , Reifies s W
  , KnownNetwork i hs o
  ) =>
  Pass ->
  BVar s (NeuralNetwork i hs o a) ->
  BVar s (UMat m i a) ->
  BVar s (UMat m o a)
{-# INLINE runNN #-}
runNN pass = go $ networkShape @i @hs @o
  where
    go ::
      forall h ls.
      (KnownNat h) =>
      NetworkShape h ls o ->
      BVar s (NeuralNetwork h ls o a) ->
      BVar s (UMat m h a) ->
      BVar s (UMat m o a)
    {-# INLINE go #-}
    go IsOutput _ = id
    go (IsCons rest) lay =
      let !decons = lay ^^. topLayerL
       in go rest (decons ^^. #continue) . runLayer pass (decons ^^. #topLayer)

evalBatchNN ::
  ( KnownNat m
  , U.Unbox a
  , RealFloat a
  , KnownNetwork i hs o
  , Backprop a
  ) =>
  NeuralNetwork i hs o a ->
  UMat m i a ->
  UMat m o a
{-# INLINE evalBatchNN #-}
evalBatchNN = evalBP2 $ runNN Eval

-- | The variant of 'evalBatchNN' with functorial inputs and outputs.
evalBatchF ::
  forall t u hs i o a v.
  ( U.Unbox a
  , RealFloat a
  , KnownNetwork i hs o
  , G.Vector v (t a)
  , G.Vector v (u a)
  , i ~ Size t
  , o ~ Size u
  , HasSize t
  , FromVec u
  , Backprop a
  , M.Load (VM.ARepr v) M.Ix1 (u a)
  , M.Manifest (VM.ARepr v) (u a)
  , VM.VRepr (VM.ARepr v) ~ v
  ) =>
  NeuralNetwork i hs o a ->
  v (t a) ->
  v (u a)
{-# INLINE evalBatchF #-}
evalBatchF nn inps =
  case fromBatchData inps of
    MkSomeBatch xs -> fromRowMat $ evalBatchNN nn xs

withKnownNetwork ::
  KnownNat i =>
  ( KnownNetwork i hs o =>
    NeuralNetwork i hs o a ->
    r
  ) ->
  NeuralNetwork i hs o a ->
  r
withKnownNetwork f Output = f Output
withKnownNetwork f (la :- net) =
  withKnownNetwork (f . (la :-)) net

evalNN ::
  ( KnownNat i
  , U.Unbox a
  , RealFloat a
  , Backprop a
  ) =>
  NeuralNetwork i hs o a ->
  UVec i a ->
  UVec o a
{-# INLINE evalNN #-}
evalNN = withKnownNetwork $
  \n -> computeV . columnAt @0 . evalBatchNN n . asColumn

-- | The variant of 'evalNN' with functorial inputs and outputs.
evalF ::
  forall t u hs i o a.
  ( U.Unbox a
  , RealFloat a
  , i ~ Size t
  , o ~ Size u
  , HasSize t
  , FromVec u
  , Backprop a
  ) =>
  NeuralNetwork i hs o a ->
  t a ->
  u a
{-# INLINE evalF #-}
evalF nn = fromVec . evalNN nn . toVec

mapNetwork ::
  (KnownNat i, KnownNat o) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    k l x y b
  ) ->
  Network h i ls o a ->
  Network k i ls o b
{-# INLINE mapNetwork #-}
mapNetwork _ Output = Output
mapNetwork f (hfka :- net') = f hfka :- mapNetwork f net'

htraverseNetwork ::
  (KnownNat i, KnownNat o, Applicative f) =>
  ( forall l x y.
    (KnownNat x, KnownNat y) =>
    h l x y a ->
    f (k l x y b)
  ) ->
  Network h i ls o a ->
  f (Network k i ls o b)
{-# INLINE htraverseNetwork #-}
htraverseNetwork _ Output = pure Output
htraverseNetwork f (hfka :- net') =
  (:-) <$> f hfka <*> htraverseNetwork f net'

foldMapNetwork ::
  forall h i ls o a w.
  (KnownNat i, Monoid w) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    w
  ) ->
  Network h i ls o a ->
  w
foldMapNetwork f = go
  where
    go ::
      (KnownNat x) =>
      Network h x hs o a ->
      w
    {-# INLINE go #-}
    go Output = mempty
    go (h :- hs) = f h <> go hs

foldZipNetwork ::
  forall h g i ls o a b w.
  (KnownNat i, Monoid w) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    g l x y b ->
    w
  ) ->
  Network h i ls o a ->
  Network g i ls o b ->
  w
foldZipNetwork f = go
  where
    go ::
      (KnownNat x) =>
      Network h x hs o a ->
      Network g x hs o b ->
      w
    {-# INLINE go #-}
    go Output Output = mempty
    go (h :- hs) (g :- gs) = f h g <> go hs gs

zipNetworkWith ::
  forall h k t i ls o a b c.
  (KnownNat i, KnownNat o) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    k l x y b ->
    t l x y c
  ) ->
  Network h i ls o a ->
  Network k i ls o b ->
  Network t i ls o c
{-# INLINE zipNetworkWith #-}
zipNetworkWith f = go
  where
    go ::
      (KnownNat n', KnownNat m') =>
      Network h n' ls' m' a ->
      Network k n' ls' m' b ->
      Network t n' ls' m' c
    go Output Output = Output
    go (hxy :- hs) (kxy :- ks) = f hxy kxy :- go hs ks

zipNetworkWith3 ::
  forall h k t u i ls o a b c d.
  (KnownNat i, KnownNat o) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    k l x y b ->
    t l x y c ->
    u l x y d
  ) ->
  Network h i ls o a ->
  Network k i ls o b ->
  Network t i ls o c ->
  Network u i ls o d
{-# INLINE zipNetworkWith3 #-}
zipNetworkWith3 f = go
  where
    go ::
      (KnownNat n', KnownNat m') =>
      Network h n' ls' m' a ->
      Network k n' ls' m' b ->
      Network t n' ls' m' c ->
      Network u n' ls' m' d
    go Output Output Output = Output
    go (hxy :- hs) (kxy :- ks) (txy :- ts) = f hxy kxy txy :- go hs ks ts

type LossFunction m o a =
  forall s. Reifies s W => BVar s (UMat m o a) -> BVar s (UMat m o a) -> BVar s a

gradNN ::
  forall ls i o m a.
  ( RealFloat a
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNetwork i ls o
  ) =>
  -- | Loss function
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i ls o a ->
  GradStack i ls o a
gradNN loss (inps, oups) =
  toGradientStack
    . gradBP (\net -> loss (runNN Train net (auto inps)) (auto oups))

-- | O(1)
toGradientStack :: NeuralNetwork i ls o a -> GradStack i ls o a
{-# INLINE toGradientStack #-}
toGradientStack = coerce

-- | O(1)
fromGradientStack :: GradStack i ls o a -> NeuralNetwork i ls o a
{-# INLINE fromGradientStack #-}
fromGradientStack = coerce

crossEntropy ::
  forall o m a k.
  ( KnownNat o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , Floating a
  , Backprop k
  , VectorSpace k a
  ) =>
  LossFunction m o a
crossEntropy ys' ys =
  negate $
    sumS (ys * log ys' + (1 - ys) * log (1 - ys'))
      /. fromIntegral (dimVal @m)

-- | The variant of 'trainGD' which accepts functorial inputs and outputs.
trainGDF ::
  forall i o a hs.
  ( U.Unbox a
  , U.Unbox (i a)
  , U.Unbox (o a)
  , HasSize i
  , HasSize o
  , Backprop a
  ) =>
  RealFloat a =>
  a ->
  Int ->
  (forall m. KnownNat m => LossFunction m (Size o) a) ->
  U.Vector (i a, o a) ->
  NeuralNetwork (Size i) hs (Size o) a ->
  NeuralNetwork (Size i) hs (Size o) a
{-# INLINE trainGDF #-}
trainGDF gamma n loss dataSet =
  case fromBatchData $ pairUV dataSet of
    MkSomeBatch umats ->
      let (ins, ous) = splitRowAt @(Size i) @(Size o) umats
       in trainGD gamma n loss (computeM ins, computeM ous)

pairUV ::
  U.Vector (i a, o a) ->
  U.Vector (Product i o a)
pairUV = coerce

trainGD ::
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNat i
  ) =>
  RealFloat a =>
  a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainGD gamma n loss dataSet =
  withKnownNetwork $
    trainGD_ gamma n loss dataSet

trainGD_ ::
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNetwork i hs o
  ) =>
  RealFloat a =>
  a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainGD_ gamma n loss dataSet = last . take n . iterate' step
  where
    step net =
      net - gamma .* fromGradientStack (gradNN loss dataSet net)

data AdamParams a = AdamParams {beta1, beta2, epsilon :: !a}
  deriving (Show, Eq, Ord, Generic)

-- | The variant of 'trainAdam' with functorial inputs and outputs.
trainAdamF ::
  forall i hs o a.
  ( U.Unbox a
  , Backprop a
  , HasSize i
  , HasSize o
  , U.Unbox (i a)
  , U.Unbox (o a)
  ) =>
  RealFloat a =>
  -- | Learning Rate
  a ->
  AdamParams a ->
  Int ->
  (forall m. KnownNat m => LossFunction m (Size o) a) ->
  U.Vector (i a, o a) ->
  NeuralNetwork (Size i) hs (Size o) a ->
  NeuralNetwork (Size i) hs (Size o) a
trainAdamF gamma ap n loss dataSet =
  case fromBatchData $ pairUV dataSet of
    MkSomeBatch umats ->
      let (ins, ous) = splitRowAt @(Size i) @(Size o) umats
       in trainAdam gamma ap n loss (computeM ins, computeM ous)

trainAdam ::
  forall m i hs o a.
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNat i
  ) =>
  RealFloat a =>
  -- | Learning Rate
  a ->
  AdamParams a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainAdam gamma ap n loss dataSet =
  withKnownNetwork $
    trainAdam_ gamma ap n loss dataSet

trainAdam_ ::
  forall m i hs o a.
  ( KnownNetwork i hs o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  ) =>
  RealFloat a =>
  -- | Learning Rate
  a ->
  AdamParams a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainAdam_ gamma AdamParams {..} n loss dataSet =
  SP.fst . last . take n . iterate' step . (:!: (0 :!: 0))
  where
    step (net :!: (s :!: v)) =
      let dW = fromGradientStack $ gradNN loss dataSet net
          sN = beta2 .* s + (1 - beta2) .* (dW * dW)
          vN = beta1 .* v + (1 - beta1) .* dW
          net' = net - (gamma .* vN) / sqrt (sN +. epsilon)
       in net' :!: (sN :!: vN)

randomNetwork ::
  ( RandomGenM g r m
  , KnownNat i
  , KnownNat o
  ) =>
  g ->
  Network LayerSpec i ls o Double ->
  m (NeuralNetwork i ls o Double)
randomNetwork g = htraverseNetwork $ \case
  (AffP s :: LayerSpec _ i _ _) -> do
    ws <- replicateMatA $ normal 0.0 s g
    bs <- replicateVecA $ standard g
    pure $ Affine ws bs
  (LinP s :: LayerSpec _ i _ _) -> do
    ws <- replicateMatA $ normal 0.0 s g
    pure $ Linear ws
  (ActP :: LayerSpec _ _ _ _) -> pure $ Activate sActivation
  (BNP s :: LayerSpec _ i _ _) -> do
    sigma <- replicateVecA $ normal 0.0 s g
    mu <- replicateVecA $ normal 0.0 s g
    w <- replicateVecA $ normal 0.0 s g
    b <- replicateVecA $ normal 0.0 s g
    pure $ BatchNorm sigma mu w b

data SomeNetwork h i o a where
  MkSomeNetwork :: Network h i hs o a -> SomeNetwork h i o a

simpleSomeNetwork ::
  forall i o a.
  (KnownNat i, KnownNat o, Floating a) =>
  [(Natural, Activation)] ->
  SomeNetwork LayerSpec i o a
{-# INLINE simpleSomeNetwork #-}
simpleSomeNetwork = go
  where
    go ::
      forall k.
      KnownNat k =>
      [(Natural, Activation)] ->
      SomeNetwork LayerSpec k o a
    go [] = MkSomeNetwork $ AffP s :- ActP @( 'Sigmoid) :- Output
      where
        !s = recip $ sqrt $ fromIntegral $ dimVal @k
    go ((n, act) : rest) = case (someNatVal n, someActivation act) of
      (SomeNat (_ :: Proxy n), MkSomeActivation (sact :: SActivation act)) ->
        case go rest of
          MkSomeNetwork net ->
            let s = case sact of
                  SReLU ->
                    sqrt $ 2 / fromIntegral (dimVal @k)
                  _ ->
                    recip $ sqrt $ fromIntegral $ dimVal @k
             in MkSomeNetwork $ affine @n s :- ActP @act :- net

withSimpleNetwork ::
  (KnownNat i, KnownNat o, Floating a) =>
  [(Natural, Activation)] ->
  (forall hs. Network LayerSpec i hs o a -> r) ->
  r
{-# INLINE withSimpleNetwork #-}
withSimpleNetwork hiddens f = case simpleSomeNetwork hiddens of
  MkSomeNetwork net -> f net

data NetworkStat = NetworkStat {parameters :: Sum Int, layers :: DL.DList LayerInfo}
  deriving (Show, Eq, Ord, Generic)
  deriving (Semigroup, Monoid) via Generically NetworkStat

data LayerInfo = AffL !Int | LinL !Int | ActL Activation | BatchL !Int
  deriving (Show, Eq, Ord, Generic)

networkStat :: KnownNat i => NeuralNetwork i hs o a -> NetworkStat
networkStat = foldMapNetwork $ \case
  (Affine {} :: Layer _ i o a) ->
    NetworkStat
      { parameters = Sum $ dimVal @i * dimVal @o + dimVal @o
      , layers = DL.singleton $ AffL $ dimVal @o
      }
  (Linear {} :: Layer _ i o a) ->
    NetworkStat
      { parameters = Sum $ dimVal @i * dimVal @o
      , layers = DL.singleton $ LinL $ dimVal @o
      }
  (Activate sact :: Layer _ i o a) ->
    mempty {layers = DL.singleton $ ActL $ activationVal sact}
  (BatchNorm {} :: Layer _ i o a) ->
    NetworkStat
      { parameters = Sum $ 4 * dimVal @o
      , layers = DL.singleton $ BatchL $ dimVal @o
      }
