{-# HLINT ignore "Redundant bracket" #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
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
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module DeepLearning.NeuralNetowrk.Massiv
  ( -- * Central data-types
    NeuralNetwork,
    SomeNeuralNetwork (..),
    SomeNetwork (..),
    simpleSomeNetwork,
    activationVal,
    withSimpleNetwork,
    Network (..),
    RecParams (..),
    Weights (..),
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
    softmax_,
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
    SLayerKind (..),
    KnownLayerKind (..),
    LayerLike,
    Aff,
    Lin,
    Act,
    BN,

    -- * Operators for manipulating networks
    withKnownNeuralNetwork,
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

import Control.Applicative (liftA2)
import Control.Arrow ((>>>))
import Control.DeepSeq (NFData (..))
import Control.Lens (Lens', lens, _1, _2)
import Control.Monad (when)
import Control.Subcategory.Linear
import Data.Bifunctor (bimap)
import qualified Data.Bifunctor as Bi
import Data.Coerce (coerce)
import qualified Data.DList as DL
import Data.Functor.Product (Product)
import Data.Generics.Labels ()
import Data.Kind (Type)
import Data.List (iterate')
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array.Manifest.Vector as VM
import Data.Monoid (Ap (..), Sum (..))
import Data.Persist (Get, Persist (..), Put)
import Data.Proxy (Proxy (..))
import Data.Strict (Pair (..))
import qualified Data.Strict.Tuple as SP
import Data.Tuple (swap)
import Data.Type.Equality (testEquality, type (:~:) (..))
import Data.Type.Natural
import qualified Data.Vector.Generic as G
import Data.Vector.Orphans
import qualified Data.Vector.Unboxed as U
import Data.Word (Word8)
import Generic.Data
import Numeric.Backprop
import Numeric.Function.Activation (relu, sigmoid, softmax)
import Numeric.Natural (Natural)
import System.Random.MWC.Distributions (normal)
import System.Random.Stateful
import Type.Reflection (typeRep)

data LayerKind = Aff | Lin | Act Activation | BN
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (Persist)

data Activation = ReLU | Sigmoid | Tanh | Softmax | Id
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)
  deriving anyclass (Persist)

data SActivation (a :: Activation) where
  SReLU :: SActivation 'ReLU
  SSigmoid :: SActivation 'Sigmoid
  STanh :: SActivation 'Tanh
  SSoftmax :: SActivation 'Softmax
  SId :: SActivation 'Id

instance KnownActivation a => Persist (SActivation a) where
  put = const $ pure ()
  {-# INLINE put #-}
  get = pure sActivation
  {-# INLINE get #-}

instance Persist SomeActivation where
  put = \case (MkSomeActivation v) -> put $ activationVal v
  {-# INLINE put #-}
  get = someActivation <$> get
  {-# INLINE get #-}

data SomeActivation where
  MkSomeActivation :: KnownActivation act => SActivation act -> SomeActivation

deriving instance Show SomeActivation

activationVal :: SActivation act -> Activation
activationVal SReLU = ReLU
activationVal SSigmoid = Sigmoid
activationVal STanh = Tanh
activationVal SSoftmax = Softmax
activationVal SId = Id

someActivation :: Activation -> SomeActivation
someActivation ReLU = MkSomeActivation SReLU
someActivation Sigmoid = MkSomeActivation SSigmoid
someActivation Tanh = MkSomeActivation STanh
someActivation Softmax = MkSomeActivation SSoftmax
someActivation Id = MkSomeActivation SId

deriving instance Show (SActivation a)

class KnownActivation (a :: Activation) where
  sActivation :: SActivation a

instance KnownActivation 'ReLU where
  sActivation = SReLU

instance KnownActivation 'Sigmoid where
  sActivation = SSigmoid

instance KnownActivation 'Tanh where
  sActivation = STanh

instance KnownActivation 'Softmax where
  sActivation = SSoftmax

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

type LayerSpec :: LayerLike
data LayerSpec l n m a where
  AffP :: a -> LayerSpec 'Aff n m a
  LinP :: a -> LayerSpec 'Lin n m a
  ActP :: KnownActivation act => LayerSpec ( 'Act act) n n a
  BNP :: LayerSpec 'BN n n a

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

softmax_ :: forall i a. LayerSpec (Act 'Softmax) i i a
softmax_ = ActP

passthru_ :: forall i a. LayerSpec (Act 'Id) i i a
passthru_ = ActP

batchnorm :: forall i a. LayerSpec 'BN i i a
batchnorm = BNP

type Activation' :: (Type -> Type) -> LayerKind -> Nat -> Nat -> Type -> Type
data Activation' m l i o a = Activation' {getActivation :: Activation, seed :: m a}
  deriving (Show, Eq, Ord, Generic, Bounded, Functor, Foldable, Traversable)

-- | Weights optimised by backprop.
type Weights :: LayerKind -> Nat -> Nat -> Type -> Type
data Weights l i o a where
  AffW :: !(UMat i o a) -> !(UVec o a) -> Weights 'Aff i o a
  LinW :: !(UMat i o a) -> Weights 'Lin i o a
  ActW :: Weights ( 'Act act) i i a
  BatW :: {scale, shift :: !(UVec i a)} -> Weights 'BN i i a

liftBinWs ::
  (UMat i o a -> UMat i o a -> UMat i o a) ->
  (UVec o a -> UVec o a -> UVec o a) ->
  Weights l i o a ->
  Weights l i o a ->
  Weights l i o a
{-# INLINE liftBinWs #-}
liftBinWs fW fV (AffW w b) (AffW w' b') = AffW (fW w w') (fV b b')
liftBinWs fW _ (LinW w) (LinW w') = LinW (fW w w')
liftBinWs _ _ l@ActW ActW {} = l
liftBinWs _ fV (BatW w b) (BatW w' b') = BatW (fV w w') (fV b b')

liftUnWs ::
  (UMat i o a -> UMat i o a) ->
  (UVec o a -> UVec o a) ->
  Weights l i o a ->
  Weights l i o a
{-# INLINE liftUnWs #-}
liftUnWs fW fV (AffW w b) = AffW (fW w) (fV b)
liftUnWs fW _ (LinW w) = LinW (fW w)
liftUnWs _ _ l@ActW = l
liftUnWs _ fV (BatW scale shift) = BatW (fV scale) (fV shift)

instance (KnownLayerKind l i o, U.Unbox a, Floating a) => Num (Weights l i o a) where
  (+) = liftBinWs (+) (+)
  {-# INLINE (+) #-}
  (-) = liftBinWs (-) (-)
  {-# INLINE (-) #-}
  (*) = liftBinWs (*) (*)
  {-# INLINE (*) #-}
  abs = liftUnWs abs abs
  {-# INLINE abs #-}
  signum = liftUnWs signum signum
  {-# INLINE signum #-}
  fromInteger = reps . fromInteger
  {-# INLINE fromInteger #-}
  negate = liftUnWs negate negate
  {-# INLINE negate #-}

instance (KnownLayerKind l i o, U.Unbox a, Floating a) => Fractional (Weights l i o a) where
  fromRational = reps . fromRational
  {-# INLINE fromRational #-}
  recip = liftUnWs recip recip
  {-# INLINE recip #-}
  (/) = liftBinWs (/) (/)
  {-# INLINE (/) #-}

instance
  (KnownLayerKind l i o, U.Unbox a, Floating a) =>
  Floating (Weights l i o a)
  where
  pi = reps pi
  {-# INLINE pi #-}
  exp = liftUnWs exp exp
  {-# INLINE exp #-}
  log = liftUnWs log log
  {-# INLINE log #-}
  sin = liftUnWs sin sin
  {-# INLINE sin #-}
  cos = liftUnWs cos cos
  {-# INLINE cos #-}
  tan = liftUnWs tan tan
  {-# INLINE tan #-}
  asin = liftUnWs asin asin
  {-# INLINE asin #-}
  acos = liftUnWs acos acos
  {-# INLINE acos #-}
  atan = liftUnWs atan atan
  {-# INLINE atan #-}
  sinh = liftUnWs sinh sinh
  {-# INLINE sinh #-}
  cosh = liftUnWs cosh cosh
  {-# INLINE cosh #-}
  tanh = liftUnWs tanh tanh
  {-# INLINE tanh #-}
  asinh = liftUnWs asinh asinh
  {-# INLINE asinh #-}
  acosh = liftUnWs acosh acosh
  {-# INLINE acosh #-}
  atanh = liftUnWs atanh atanh
  {-# INLINE atanh #-}

instance
  (KnownLayerKind l i o, U.Unbox a, Floating a) =>
  VectorSpace a (Weights l i o a)
  where
  reps = case sLayerKind @l @i @o of
    SAff -> AffW <$> reps <*> reps
    SLin -> LinW <$> reps
    SAct _ -> const ActW
    SBN -> BatW <$> reps <*> reps
  {-# INLINE reps #-}
  (.*) = liftUnWs <$> (.*) <*> (.*)
  {-# INLINE (.*) #-}
  (*.) = flip $ liftUnWs <$> flip (*.) <*> flip (*.)
  {-# INLINE (*.) #-}
  (.+) = liftUnWs <$> (.+) <*> (.+)
  {-# INLINE (.+) #-}
  (+.) = flip $ liftUnWs <$> flip (+.) <*> flip (+.)
  {-# INLINE (+.) #-}
  (.-) = liftUnWs <$> (.-) <*> (.-)
  {-# INLINE (.-) #-}
  (-.) = flip $ liftUnWs <$> flip (-.) <*> flip (-.)
  {-# INLINE (-.) #-}
  (/.) = flip $ liftUnWs <$> flip (/.) <*> flip (/.)
  {-# INLINE (/.) #-}
  AffW w b >.< AffW w' b' = w >.< w' + b >.< b'
  LinW w >.< LinW w' = w >.< w'
  ActW >.< ActW {} = 0
  BatW w b >.< BatW w' b' = w >.< w' + b >.< b'
  {-# INLINE (>.<) #-}
  sumS (AffW w b) = sumS w + sumS b
  sumS (LinW w) = sumS w
  sumS ActW = 0
  sumS (BatW mu sigma) = sumS mu + sumS sigma
  {-# INLINE sumS #-}

-- | Parameters updated by step-by-step, but not optimised by backprop.
type RecParams :: LayerKind -> Nat -> Nat -> Type -> Type
data RecParams l i o a where
  AffRP :: RecParams 'Aff i o a
  LinRP :: RecParams 'Lin i o a
  ActRP :: !(SActivation act) -> RecParams ( 'Act act) i i a
  BatRP :: {mean, deviation :: !(UVec i a)} -> RecParams 'BN i i a

liftBinRP ::
  (UVec o a -> UVec o a -> UVec o a) ->
  RecParams l i o a ->
  RecParams l i o a ->
  RecParams l i o a
{-# INLINE liftBinRP #-}
liftBinRP _ l@AffRP AffRP {} = l
liftBinRP _ l@LinRP LinRP {} = l
liftBinRP _ l@(ActRP _) ActRP {} = l
liftBinRP fV (BatRP mu sigma) (BatRP mu' sigma') =
  BatRP (fV mu mu') (fV sigma sigma')

liftUnRP ::
  (UVec o a -> UVec o a) ->
  RecParams l i o a ->
  RecParams l i o a
{-# INLINE liftUnRP #-}
liftUnRP _ l@AffRP = l
liftUnRP _ l@LinRP = l
liftUnRP _ l@(ActRP _) = l
liftUnRP fV (BatRP mu sigma) = BatRP (fV mu) (fV sigma)

instance (KnownLayerKind l i o, U.Unbox a, Floating a) => Num (RecParams l i o a) where
  (+) = liftBinRP (+)
  {-# INLINE (+) #-}
  (-) = liftBinRP (-)
  {-# INLINE (-) #-}
  (*) = liftBinRP (*)
  {-# INLINE (*) #-}
  abs = liftUnRP abs
  {-# INLINE abs #-}
  signum = liftUnRP signum
  {-# INLINE signum #-}
  fromInteger = reps . fromInteger
  {-# INLINE fromInteger #-}
  negate = liftUnRP negate
  {-# INLINE negate #-}

instance
  (KnownLayerKind l i o, U.Unbox a, Floating a) =>
  VectorSpace a (RecParams l i o a)
  where
  reps = case sLayerKind @l @i @o of
    SAff -> const AffRP
    SLin -> const LinRP
    SAct sa -> const $ ActRP sa
    SBN -> BatRP <$> reps <*> reps
  {-# INLINE reps #-}
  (.*) = liftUnRP <$> (.*)
  {-# INLINE (.*) #-}
  (*.) = flip $ liftUnRP <$> flip (*.)
  {-# INLINE (*.) #-}
  (.+) = liftUnRP <$> (.+)
  {-# INLINE (.+) #-}
  (+.) = flip $ liftUnRP <$> flip (+.)
  {-# INLINE (+.) #-}
  (.-) = liftUnRP <$> (.-)
  {-# INLINE (.-) #-}
  (-.) = flip $ liftUnRP <$> flip (-.)
  {-# INLINE (-.) #-}
  (/.) = flip $ liftUnRP <$> flip (/.)
  {-# INLINE (/.) #-}
  AffRP >.< AffRP {} = 0
  LinRP >.< LinRP {} = 0
  ActRP _ >.< ActRP {} = 0
  BatRP mu sigma >.< BatRP mu' sigma' = mu >.< mu' + sigma >.< sigma'
  {-# INLINE (>.<) #-}
  sumS AffRP = 0
  sumS LinRP = 0
  sumS (ActRP _) = 0
  sumS (BatRP mu sigma) = sumS mu + sumS sigma
  {-# INLINE sumS #-}

instance (KnownLayerKind l i o, U.Unbox a, Floating a) => Fractional (RecParams l i o a) where
  fromRational = reps . fromRational
  {-# INLINE fromRational #-}
  recip = liftUnRP recip
  {-# INLINE recip #-}
  (/) = liftBinRP (/)
  {-# INLINE (/) #-}

instance
  (KnownLayerKind l i o, U.Unbox a, Floating a) =>
  Floating (RecParams l i o a)
  where
  pi = reps pi
  {-# INLINE pi #-}
  exp = liftUnRP exp
  {-# INLINE exp #-}
  log = liftUnRP log
  {-# INLINE log #-}
  sin = liftUnRP sin
  {-# INLINE sin #-}
  cos = liftUnRP cos
  {-# INLINE cos #-}
  tan = liftUnRP tan
  {-# INLINE tan #-}
  asin = liftUnRP asin
  {-# INLINE asin #-}
  acos = liftUnRP acos
  {-# INLINE acos #-}
  atan = liftUnRP atan
  {-# INLINE atan #-}
  sinh = liftUnRP sinh
  {-# INLINE sinh #-}
  cosh = liftUnRP cosh
  {-# INLINE cosh #-}
  tanh = liftUnRP tanh
  {-# INLINE tanh #-}
  asinh = liftUnRP asinh
  {-# INLINE asinh #-}
  acosh = liftUnRP acosh
  {-# INLINE acosh #-}
  atanh = liftUnRP atanh
  {-# INLINE atanh #-}

instance
  ( KnownNat o
  , U.Unbox a
  , Floating a
  ) =>
  Backprop (RecParams l i o a)
  where
  zero = \case
    l@AffRP -> l
    l@LinRP -> l
    l@(ActRP _) -> l
    BatRP l r -> BatRP (zero l) (zero r)
  one = \case
    AffRP -> AffRP
    LinRP -> LinRP
    ActRP s -> ActRP s
    BatRP l r -> BatRP (one l) (one r)
  add l@AffRP AffRP {} = l
  add l@LinRP LinRP {} = l
  add l@(ActRP _) ActRP {} = l
  add (BatRP mu sigma) (BatRP mu' sigma') =
    BatRP (add mu mu') (add sigma sigma')
  {-# INLINE add #-}

instance (forall l x y. Semigroup (h l x y a)) => Semigroup (Network h i ls o a) where
  Output <> Output = Output
  (a :- as) <> (b :- bs) = (a <> b) :- (as <> bs)
  {-# INLINE (<>) #-}

affineMatL :: Lens' (Weights Aff i o a) (UMat i o a)
{-# INLINE affineMatL #-}
affineMatL = lens (\case (AffW mat _) -> mat) $
  \case (AffW _ vec) -> (`AffW` vec)

affineBiasL :: Lens' (Weights Aff i o a) (UVec o a)
{-# INLINE affineBiasL #-}
affineBiasL = lens (\case (AffW _ v) -> v) $
  \case (AffW mat _) -> AffW mat

linearMatL :: Lens' (Weights Lin i o a) (UMat i o a)
{-# INLINE linearMatL #-}
linearMatL = lens (\case (LinW mat) -> mat) $ const LinW

data BatchLayer i a = BatchLayer {scale, shift :: UVec i a}
  deriving (Show, Generic)

deriving anyclass instance
  (KnownNat i, M.Numeric M.U a, U.Unbox a) => Backprop (BatchLayer i a)

batchNormL :: Lens' (Weights BN i i a) (BatchLayer i a)
batchNormL =
  lens
    (\BatW {..} -> BatchLayer {..})
    $ const $ \BatchLayer {..} -> BatW {..}

data Pass = Train | Eval
  deriving (Show, Eq, Ord, Generic, Bounded, Enum)

runLayer ::
  forall s m l a i o.
  ( U.Unbox a
  , KnownNat i
  , KnownNat o
  , KnownNat m -- batch size
  , RealFloat a
  , Backprop a
  , Reifies s W
  ) =>
  Pass ->
  RecParams l i o a ->
  BVar s (Weights l i o a) ->
  BVar s (UMat m i a) ->
  BVar s (UMat m o a, RecParams l i o a)
{-# INLINE runLayer #-}
runLayer pass = \case
  l@AffRP {} -> \aff x ->
    let w = aff ^^. affineMatL
        b = aff ^^. affineBiasL
     in T2 (w !*!: x + duplicateAsCols' b) (auto l)
  l@LinRP -> \lin x ->
    let w = lin ^^. linearMatL
     in T2 (w !*!: x) (auto l)
  l@(ActRP a) -> const $ flip T2 (auto l) . applyActivatorBV a
  l@BatRP {..} -> \lay ->
    let !bnParams = lay ^^. batchNormL
        !mu = mean
        !sigma = deviation
        !gamma = bnParams ^^. #scale
        !beta = bnParams ^^. #shift
     in case pass of
          Train -> \x ->
            let !m = fromIntegral $ dimVal @m
                batchMu = sumRows' x /. m
                xRel = x - duplicateAsCols' batchMu
                batchSigma = sumRows' (xRel * xRel) /. m
                ivar = sqrt $ batchSigma +. 1e-12
                gammax =
                  duplicateAsCols' gamma * xRel / duplicateAsCols' ivar
                bRP =
                  isoVar2
                    BatRP
                    (\(BatRP z s) -> (z, s))
                    batchMu
                    batchSigma
             in T2 (gammax + duplicateAsCols' beta) bRP
          Eval -> \x ->
            let eps = 1e-12
                out1 =
                  (x - auto (computeM $ duplicateAsCols mu))
                    / auto (computeM (duplicateAsCols (sqrt (sigma +. eps))))
             in T2 (duplicateAsCols' gamma * out1 + duplicateAsCols' beta) (auto l)

applyActivatorBV ::
  ( Reifies s W
  , KnownNat n
  , KnownNat m
  , M.Load r M.Ix2 a
  , M.Load r M.Ix1 a
  , M.NumericFloat r a
  , RealFloat a
  , M.Manifest r a
  ) =>
  SActivation act ->
  BVar s (Mat r n m a) ->
  BVar s (Mat r n m a)
applyActivatorBV SReLU = relu
applyActivatorBV SSigmoid = sigmoid
applyActivatorBV STanh = tanh
applyActivatorBV SSoftmax = softmax
applyActivatorBV SId = id

-- | FIXME: defining on the case of @l@ would reduce runtime branching.
instance (KnownNat i, KnownNat o, U.Unbox a, Num a) => Backprop (Weights l i o a) where
  zero = \case
    AffW {} -> AffW 0 0
    LinW {} -> LinW 0
    l@ActW {} -> l
    BatW {} -> BatW 0 0
  {-# INLINE zero #-}
  one = \case
    AffW {} -> AffW 1 1
    LinW {} -> LinW 1
    l@ActW {} -> l
    BatW {} -> BatW 1 1
  {-# INLINE one #-}
  add (AffW mat vec) (AffW mat' vec') = AffW (add mat mat') (add vec vec')
  add (LinW mat) (LinW mat') = LinW (add mat mat')
  add l@ActW {} ActW {} = l
  add (BatW a b) (BatW a' b') = BatW (a + a') (b + b')
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

data NeuralNetwork i ls o a = NeuralNetwork
  { recParams :: !(Network RecParams i ls o a)
  , weights :: !(Network Weights i ls o a)
  }
  deriving (Generic)
  deriving anyclass (Persist)

data SomeNeuralNetwork i o a where
  MkSomeNeuralNetwork :: NeuralNetwork i ls o a -> SomeNeuralNetwork i o a

instance
  ( KnownNat i
  , KnownNat o
  , Persist a
  , U.Unbox a
  ) =>
  Persist (SomeNeuralNetwork i o a)
  where
  put (MkSomeNeuralNetwork NeuralNetwork {..}) =
    withKnownNetwork recParams $ do
      put $ MkSomeNetwork weights
      put recParams
  get =
    getSomeNetwork >>= \case
      MkSomeNetwork (weights :: Network Weights i hs o a) ->
        withKnownNetwork weights $ do
          recParams <- get
          pure $ MkSomeNeuralNetwork NeuralNetwork {..}

instance
  ( KnownNat i
  , KnownNat o
  , KnownNetwork i fs o
  , forall l x y. KnownLayerKind l x y => Num (h l x y a)
  ) =>
  Num (Network h i fs o a)
  where
  fromInteger = go $ networkShape @i @fs @o
    where
      go :: NetworkShape l hs o -> Integer -> Network h l hs o a
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
  , Num a
  , KnownNetwork i fs o
  , Floating a
  , forall l x y. (KnownLayerKind l x y, KnownNat x, KnownNat y) => VectorSpace a (h l x y a)
  ) =>
  VectorSpace a (Network h i fs o a)
  where
  reps = go $ networkShape @i @fs @o
    where
      go :: KnownNat l => NetworkShape l hs o -> a -> Network h l hs o a
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
  , KnownNetwork i fs o
  , forall l x y. (KnownLayerKind l x y => Fractional (h l x y a))
  ) =>
  Fractional (Network h i fs o a)
  where
  fromRational = go $ networkShape @i @fs @o
    where
      go :: NetworkShape l hs o -> Rational -> Network h l hs o a
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
  , forall l x y. (KnownLayerKind l x y => Floating (h l x y a))
  ) =>
  Floating (Network h i fs o a)
  where
  pi = go $ networkShape @i @fs @o
    where
      go :: NetworkShape l hs o -> Network h l hs o a
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

networkSize :: NetworkShape i xs o -> Word
networkSize IsOutput = 0
networkSize (IsCons n) = 1 + networkSize n

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
  , forall l' x y. (KnownNat x, KnownNat y) => Backprop (h l' x y a)
  ) =>
  Backprop (TopLayerView h i l k xs o a)

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
  , (forall l x y. ((KnownNat x, KnownNat y) => Backprop (h l x y a)))
  ) =>
  Backprop (Network h i ls o a)
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
  , KnownNat i
  , KnownNat o
  , Reifies s W
  ) =>
  Pass ->
  Network RecParams i hs o a ->
  BVar s (Network Weights i hs o a) ->
  BVar s (UMat m i a) ->
  BVar s (UMat m o a, Network RecParams i hs o a)
{-# INLINE runNN #-}
runNN pass = go
  where
    go ::
      forall h ls.
      (KnownNat h) =>
      Network RecParams h ls o a ->
      BVar s (Network Weights h ls o a) ->
      BVar s (UMat m h a) ->
      BVar s (UMat m o a, Network RecParams h ls o a)
    {-# INLINE go #-}
    go Output _ = flip T2 (auto Output)
    go (ps :- restPs) lay = \x ->
      let !decons = lay ^^. topLayerL
          !top = decons ^^. #topLayer
          !vLays' = runLayer pass ps top x
          !passed = go restPs (decons ^^. #continue) (vLays' ^^. _1)
          !v' = passed ^^. _1
          !rest' = passed ^^. _2
       in T2 v' (isoVar2 (:-) (\(a :- b) -> (a, b)) (vLays' ^^. _2) rest')

curryNN ::
  (Network RecParams i ls o a -> Network Weights i ls o a -> r) ->
  NeuralNetwork i ls o a ->
  r
curryNN f = f <$> recParams <*> weights

evalBatchNN ::
  ( KnownNat m
  , U.Unbox a
  , RealFloat a
  , Backprop a
  , KnownNat i
  , KnownNat o
  ) =>
  NeuralNetwork i hs o a ->
  UMat m i a ->
  UMat m o a
{-# INLINE evalBatchNN #-}
evalBatchNN = curryNN $ \ps ->
  evalBP2 (fmap (viewVar _1) . runNN Eval ps)

-- | The variant of 'evalBatchNN' with functorial inputs and outputs.
evalBatchF ::
  forall t u hs i o a v v'.
  ( U.Unbox a
  , RealFloat a
  , G.Vector v (t a)
  , i ~ Size t
  , o ~ Size u
  , HasSize t
  , FromVec u
  , Backprop a
  , VM.VRepr (VM.ARepr v') ~ v'
  , M.Load (VM.ARepr v') M.Ix1 (u a)
  , M.Manifest (VM.ARepr v') (u a)
  , G.Vector v' (u a)
  ) =>
  NeuralNetwork i hs o a ->
  v (t a) ->
  v' (u a)
{-# INLINE evalBatchF #-}
evalBatchF nn inps =
  case fromBatchData inps of
    MkSomeBatch xs -> fromRowMat $ evalBatchNN nn xs

withKnownNeuralNetwork ::
  KnownNat i =>
  ( KnownNetwork i hs o =>
    NeuralNetwork i hs o a ->
    r
  ) ->
  NeuralNetwork i hs o a ->
  r
withKnownNeuralNetwork f n = withKnownNetwork (recParams n) (f n)

withKnownNetwork ::
  KnownNat i =>
  Network h i hs o a ->
  ( KnownNetwork i hs o =>
    r
  ) ->
  r
withKnownNetwork Output f = f
withKnownNetwork (_ :- ps) f = withKnownNetwork ps f

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
evalNN = withKnownNeuralNetwork $
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
  (Monoid w) =>
  ( forall l x y.
    (KnownLayerKind l x y) =>
    h l x y a ->
    w
  ) ->
  Network h i ls o a ->
  w
foldMapNetwork f = go
  where
    go :: Network h x hs o a -> w
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
  forall ls i o m a k.
  ( RealFloat a
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNat i
  , KnownNat o
  , Backprop k
  , VectorSpace k a
  ) =>
  -- | Loss function
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  Network RecParams i ls o a ->
  Network Weights i ls o a ->
  (Network Weights i ls o a, Network RecParams i ls o a)
gradNN loss (inps, oups) recPs =
  bimap ($ (1, zero recPs)) snd
    . swap
    . backpropWith
      ( \net ->
          let ysRecs' = runNN Train recPs net (auto inps)
           in T2 (loss (ysRecs' ^^. _1) (auto oups) /. fromIntegral (dimVal @m)) (ysRecs' ^^. _2)
      )

crossEntropy ::
  forall o m a.
  ( KnownNat o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , Floating a
  ) =>
  LossFunction m o a
crossEntropy ys' ys =
  negate $
    sumS (ys * log ys' + (1 - ys) * log (1 - ys'))

-- | The variant of 'trainGD' which accepts functorial inputs and outputs.
trainGDF ::
  forall i o a hs k.
  ( U.Unbox a
  , U.Unbox (i a)
  , U.Unbox (o a)
  , HasSize i
  , HasSize o
  , Backprop a
  , VectorSpace k a
  , Backprop k
  ) =>
  RealFloat a =>
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  Int ->
  (forall m. KnownNat m => LossFunction m (Size o) a) ->
  U.Vector (i a, o a) ->
  NeuralNetwork (Size i) hs (Size o) a ->
  NeuralNetwork (Size i) hs (Size o) a
{-# INLINE trainGDF #-}
trainGDF gamma alpha n loss dataSet =
  case fromBatchData $ pairUV dataSet of
    MkSomeBatch umats ->
      let (ins, ous) = splitRowAt @(Size i) @(Size o) umats
       in trainGD gamma alpha n loss (computeM ins, computeM ous)

pairUV ::
  U.Vector (i a, o a) ->
  U.Vector (Product i o a)
pairUV = coerce

trainGD ::
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , VectorSpace k a
  , Backprop k
  , KnownNat i
  ) =>
  RealFloat a =>
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainGD gamma alpha n loss dataSet =
  withKnownNeuralNetwork $
    trainGD_ gamma alpha n loss dataSet

trainGD_ ::
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , VectorSpace k a
  , Backprop k
  , KnownNetwork i hs o
  ) =>
  RealFloat a =>
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainGD_ gamma alpha n loss dataSet = last . take n . iterate' step
  where
    step (NeuralNetwork ps ws) =
      let (ws', ps') = gradNN loss dataSet ps ws
       in NeuralNetwork (alpha .* ps' + (1 - alpha) .* ps) (ws - gamma .* ws')

data AdamParams a = AdamParams {beta1, beta2, epsilon :: !a}
  deriving (Show, Eq, Ord, Generic)

-- | The variant of 'trainAdam' with functorial inputs and outputs.
trainAdamF ::
  forall i hs o a k.
  ( U.Unbox a
  , Backprop a
  , HasSize i
  , HasSize o
  , U.Unbox (i a)
  , U.Unbox (o a)
  , VectorSpace k a
  , Backprop k
  ) =>
  RealFloat a =>
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  AdamParams a ->
  Int ->
  (forall m. KnownNat m => LossFunction m (Size o) a) ->
  U.Vector (i a, o a) ->
  NeuralNetwork (Size i) hs (Size o) a ->
  NeuralNetwork (Size i) hs (Size o) a
trainAdamF gamma alpha ap n loss dataSet =
  case fromBatchData $ pairUV dataSet of
    MkSomeBatch umats ->
      let (ins, ous) = splitRowAt @(Size i) @(Size o) umats
       in trainAdam gamma alpha ap n loss (computeM ins, computeM ous)

trainAdam ::
  forall m i hs o a k.
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNat i
  , VectorSpace k a
  , Backprop k
  ) =>
  RealFloat a =>
  -- | Learning rate (dt)
  a ->
  -- | Dumping factor for batchnorm
  a ->
  AdamParams a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainAdam gamma alpha ap n loss dataSet =
  withKnownNeuralNetwork $
    trainAdam_ gamma alpha ap n loss dataSet

trainAdam_ ::
  forall m i hs o a k.
  ( KnownNetwork i hs o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , VectorSpace k a
  , Backprop k
  ) =>
  RealFloat a =>
  -- | Learning Rate (dt)
  a ->
  -- | Dumping factor
  a ->
  AdamParams a ->
  Int ->
  LossFunction m o a ->
  (UMat m i a, UMat m o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainAdam_ gamma alpha AdamParams {..} n loss dataSet =
  SP.fst . last . take n . iterate' step . (:!: (0 :!: 0))
  where
    step ((NeuralNetwork ps net) :!: (s :!: v)) =
      let (dW, ps') = gradNN loss dataSet ps net
          sN = beta2 .* s + (1 - beta2) .* (dW * dW)
          vN = beta1 .* v + (1 - beta1) .* dW
          net' = net - (gamma .* vN) / sqrt (sN +. epsilon)
          ps'' = alpha .* ps' + (1 - alpha) .* ps
       in NeuralNetwork ps'' net' :!: (sN :!: vN)

randomNetwork ::
  ( RandomGenM g r m
  , KnownNat i
  , KnownNat o
  ) =>
  g ->
  Network LayerSpec i ls o Double ->
  m (NeuralNetwork i ls o Double)
randomNetwork g =
  liftA2 NeuralNetwork
    <$> htraverseNetwork \case
      (AffP _ :: LayerSpec _ i _ _) -> pure AffRP
      (LinP _ :: LayerSpec _ i _ _) -> pure LinRP
      (ActP :: LayerSpec _ _ _ _) -> pure $ ActRP sActivation
      (BNP :: LayerSpec _ i _ _) -> pure $ BatRP 0 1
    <*> htraverseNetwork \case
      (AffP s :: LayerSpec _ i _ _) -> do
        ws <- replicateMatA $ normal 0.0 s g
        pure $ AffW ws 0.0
      (LinP s :: LayerSpec _ i _ _) -> do
        ws <- replicateMatA $ normal 0.0 s g
        pure $ LinW ws
      (ActP :: LayerSpec _ _ _ _) -> pure ActW
      (BNP :: LayerSpec _ i _ _) -> pure $ BatW 1 0

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
networkStat =
  recParams >>> foldMapNetwork \case
    (AffRP {} :: RecParams _ i o a) ->
      NetworkStat
        { parameters = Sum $ dimVal @i * dimVal @o + dimVal @o
        , layers = DL.singleton $ AffL $ dimVal @o
        }
    (LinRP {} :: RecParams _ i o a) ->
      NetworkStat
        { parameters = Sum $ dimVal @i * dimVal @o
        , layers = DL.singleton $ LinL $ dimVal @o
        }
    (ActRP sact :: RecParams _ i o a) ->
      mempty {layers = DL.singleton $ ActL $ activationVal sact}
    (BatRP {} :: RecParams _ i o a) ->
      NetworkStat
        { parameters = Sum $ 4 * dimVal @o
        , layers = DL.singleton $ BatchL $ dimVal @o
        }

instance
  ( KnownNat o
  , NFData a
  , U.Unbox a
  , forall l x y. NFData (h l x y a)
  ) =>
  NFData (Network h i ls o a)
  where
  rnf Output = ()
  rnf (h :- hs) = rnf h `seq` rnf hs
  {-# INLINE rnf #-}

instance NFData (Weights l i o a) where
  rnf (AffW mat vec) = rnf mat `seq` rnf vec
  rnf (LinW mat) = rnf mat
  rnf ActW = ()
  rnf (BatW vec vec') = rnf vec `seq` rnf vec'
  {-# INLINE rnf #-}

instance NFData (SActivation a) where
  rnf SReLU = ()
  rnf SSigmoid = ()
  rnf STanh = ()
  rnf SSoftmax = ()
  rnf SId = ()

instance NFData (RecParams l i o a) where
  rnf AffRP = ()
  rnf LinRP = ()
  rnf (ActRP sa) = rnf sa
  rnf (BatRP vec vec') = rnf vec `seq` rnf vec'
  {-# INLINE rnf #-}

getNetworkWith ::
  ( Persist a
  , U.Unbox a
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  ) =>
  NetworkShape i xs o ->
  Get (Network h i xs o a)
getNetworkWith shape = do
  validateNetworkHeader shape
  getNetworkBodyWith shape

validateNetworkHeader :: NetworkShape i xs o -> Get ()
validateNetworkHeader shape = do
  let !expSize = networkSize shape
  size <- get
  when (size /= expSize) $
    fail $
      "validateNetworkHeader: Network has different size; expected: "
        <> show expSize
        <> ", but got: "
        <> show size

getNetworkBodyWith ::
  ( Persist a
  , U.Unbox a
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  ) =>
  NetworkShape i xs o ->
  Get (Network h i xs o a)
getNetworkBodyWith IsOutput = pure Output
getNetworkBodyWith (IsCons ns') = (:-) <$> get <*> getNetworkWith ns'

putNetworkWithHeader ::
  ((forall l x y. KnownLayerKind l x y => Persist (h l x y a))) =>
  Network h i xs o a ->
  Put ()
putNetworkWithHeader =
  SP.uncurry (*>)
    . Bi.bimap (put . getSum @Word) getAp
    . foldMapNetwork ((1 :!:) <$> Ap . put)

data SomeLayer h a where
  MkSomeLayer :: KnownLayerKind l x y => h l x y a -> SomeLayer h a

instance
  ( forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  , Persist (SomeLayer h a)
  , KnownNat i
  , KnownNat o
  ) =>
  Persist (SomeNetwork h i o a)
  where
  put (MkSomeNetwork net) = putNetworkWithHeader net
  {-# INLINE put #-}
  get = getSomeNetwork
  {-# INLINE get #-}

getSomeNetwork ::
  forall h i o a.
  ( Persist (SomeLayer h a)
  , KnownNat i
  , KnownNat o
  ) =>
  Get (SomeNetwork h i o a)
getSomeNetwork = do
  n <- get @Word
  go @i n
  where
    go :: forall n. KnownNat n => Word -> Get (SomeNetwork h n o a)
    go 0 =
      case testEquality (typeRep @n) (typeRep @o) of
        Just Refl -> pure $ MkSomeNetwork Output
        Nothing ->
          fail $
            "getSomeNetwork: output size mismatched: (expect, got) = "
              <> show (natVal @o Proxy, natVal @n Proxy)
    go !n = do
      MkSomeLayer (l :: h l n' k a) <- get
      case testEquality (typeRep @n') (typeRep @n) of
        Just Refl -> do
          MkSomeNetwork !rest <- go @k (n - 1)
          pure $ MkSomeNetwork $ l :- rest
        Nothing ->
          fail $
            "getSomeNetwork.go: input size mismatched: (exp, got) = "
              <> show (natVal @n Proxy, natVal @n' Proxy)

instance
  ( Persist a
  , U.Unbox a
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  , KnownNetwork i ls o
  ) =>
  Persist (Network h i ls o a)
  where
  put = putNetworkWithHeader
  {-# INLINE put #-}
  get = getNetworkWith networkShape
  {-# INLINE get #-}

instance
  (KnownLayerKind l i o, Persist a, U.Unbox a) =>
  Persist (Weights l i o a)
  where
  get = case sLayerKind @l @i @o of
    sa@SAff -> AffW <$ expectTag (toTag sa) <*> get <*> get
    sa@SLin -> LinW <$ expectTag (toTag sa) <*> get
    sa@(SAct _) -> ActW <$ expectTag (toTag sa) <* expectDim @i
    sa@SBN -> BatW <$ expectTag (toTag sa) <*> get <*> get
  put (AffW mat vec) =
    put Aff *> put mat *> put vec
  put (LinW mat) = put Lin *> put mat
  put ActW = case sLayerKind @l @i @o of
    SAct sa -> put (Act $ activationVal sa) *> putDim @i
  put (BatW vec vec') = put BN *> put vec *> put vec'
  {-# INLINE put #-}

instance
  (KnownLayerKind l i o, Persist a, U.Unbox a) =>
  Persist (RecParams l i o a)
  where
  get = case sLayerKind @l @i @o of
    sa@SAff -> AffRP <$ expectTag (toTag sa)
    sa@SLin -> LinRP <$ expectTag (toTag sa)
    sa@(SAct sact) -> ActRP sact <$ expectTag (toTag sa) <* expectDim @i
    sa@SBN -> BatRP <$ expectTag (toTag sa) <*> get <*> get
  put AffRP = put Aff
  put LinRP = put Lin
  put (ActRP sact) = do
    put $ Act $ activationVal sact
    putDim @i
  put (BatRP vec vec') = put BN *> put vec *> put vec'
  {-# INLINE put #-}

putDim :: forall i. KnownNat i => Put ()
putDim = put @Word $ fromIntegral $ dimVal @i

expectDim :: forall i. KnownNat i => Get ()
expectDim = do
  idx <- get @Word
  when (idx /= fromIntegral (dimVal @i)) $
    fail $ "Dimension mismatched: " <> show (dimVal @i, idx)

putSomeLayer :: (forall l x y. KnownLayerKind l x y => Persist (h l x y a)) => SomeLayer h a -> Put ()
putSomeLayer (MkSomeLayer l) = put l

instance (U.Unbox a, Persist a) => Persist (SomeLayer Weights a) where
  get =
    get >>= \case
      Aff -> do
        mat <- get
        vec <- get
        case (M.size mat, M.size vec) of
          (M.Sz2 o i, M.Sz1 o')
            | o == o' ->
              case (someNatVal (fromIntegral o), someNatVal (fromIntegral i)) of
                (SomeNat (_ :: Proxy o), SomeNat (_ :: Proxy i)) ->
                  pure $ MkSomeLayer $ AffW (unsafeToMat @i @o mat) (unsafeToVec @o vec)
            | otherwise ->
              fail $
                "SomeLayer Weight: decoding mismatched: "
                  <> show (M.size mat, M.size vec)
      Lin -> do
        mat <- get
        let M.Sz2 o i = M.size mat
        case (someNatVal (fromIntegral o), someNatVal (fromIntegral i)) of
          (SomeNat (_ :: Proxy o), SomeNat (_ :: Proxy i)) ->
            pure $ MkSomeLayer $ LinW (unsafeToMat @i @o mat)
      Act act -> do
        dim <- getDim
        case (someActivation act, someNatVal (fromIntegral dim)) of
          (MkSomeActivation (_ :: SActivation act), SomeNat (_ :: Proxy i)) ->
            pure $ MkSomeLayer $ ActW @act @i
      BN -> do
        vec <- get
        vec' <- get
        let M.Sz1 !i = M.size vec
        when (M.size vec' /= M.Sz1 i) $
          fail $ "Input size mismatched: " <> show (i, M.size vec')
        case someNatVal $ fromIntegral i of
          (SomeNat (_ :: Proxy i)) ->
            pure $ MkSomeLayer $ BatW (unsafeToVec @i vec) (unsafeToVec @i vec')
  put = putSomeLayer
  {-# INLINE put #-}

getDim :: Get Word
getDim = get

expectTag :: Word8 -> Get ()
expectTag tag = do
  tag' <- get
  when (tag /= tag') $
    fail $
      "Tag mismatched; expected: " <> show tag
        <> ", but got: "
        <> show tag'

toTag :: SLayerKind l i o -> Word8
toTag =
  \case
    SAff -> affTag
    SLin -> linTag
    SAct sact -> actTag (activationVal sact)
    SBN -> batTag

affTag, linTag, batTag :: Word8
actTag :: Activation -> Word8
affTag = 0
linTag = 1

actTag = (2 +) <$> toEnum . fromEnum

batTag = 1 + actTag maxBound
