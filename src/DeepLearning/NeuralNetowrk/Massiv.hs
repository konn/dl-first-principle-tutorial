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

{-# HLINT ignore "Redundant bracket" #-}

module DeepLearning.NeuralNetowrk.Massiv
  ( -- * Central data-types
    NeuralNetwork,
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

import Control.Applicative (liftA2)
import Control.Arrow ((&&&), (>>>))
import Control.Lens (Lens', lens, _1, _2)
import Control.Subcategory (CZip (..), cmap)
import Control.Subcategory.Linear
import Data.Bifunctor (bimap)
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
import Data.Tuple (swap)
import Data.Type.Natural
import qualified Data.Vector.Generic as G
import Data.Vector.Orphans
import qualified Data.Vector.Unboxed as U
import Generic.Data
import Numeric.Backprop
import Numeric.Function.Activation (sigmoid)
import Numeric.Natural (Natural)
import System.Random.MWC.Distributions (normal)
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
  , M.NumericFloat r a
  , RealFloat a
  , M.Manifest r a
  ) =>
  SActivation act ->
  BVar s (Mat r n m a) ->
  BVar s (Mat r n m a)
applyActivatorBV SReLU =
  liftOp1 $
    op1 (cmap (max 0) &&& czipWith (\x d -> if x < 0 then 0 else d))
applyActivatorBV SSigmoid = sigmoid
applyActivatorBV STanh = tanh
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
  , KnownNetwork i hs o
  , Backprop a
  ) =>
  NeuralNetwork i hs o a ->
  UMat m i a ->
  UMat m o a
{-# INLINE evalBatchNN #-}
evalBatchNN = curryNN $ \ps ->
  evalBP2 (fmap (viewVar _1) . runNN Eval ps)

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
withKnownNetwork f i@(NeuralNetwork Output Output) = f i
withKnownNetwork f (NeuralNetwork (ps :- pss) (ws :- wss)) =
  withKnownNetwork
    (curryNN $ \ps' ws' -> f $ NeuralNetwork (ps :- ps') (ws :- ws'))
    (NeuralNetwork pss wss)

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
  , KnownNat i
  , KnownNat o
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
           in T2 (loss (ysRecs' ^^. _1) (auto oups)) (ysRecs' ^^. _2)
      )

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
  withKnownNetwork $
    trainGD_ gamma alpha n loss dataSet

trainGD_ ::
  ( KnownNat m
  , U.Unbox a
  , Backprop a
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
  forall i hs o a.
  ( U.Unbox a
  , Backprop a
  , HasSize i
  , HasSize o
  , U.Unbox (i a)
  , U.Unbox (o a)
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
  forall m i hs o a.
  ( KnownNat m
  , U.Unbox a
  , Backprop a
  , KnownNat i
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
  withKnownNetwork $
    trainAdam_ gamma alpha ap n loss dataSet

trainAdam_ ::
  forall m i hs o a.
  ( KnownNetwork i hs o
  , KnownNat m
  , U.Unbox a
  , Backprop a
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
