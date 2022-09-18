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
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module DeepLearning.NeuralNetowrk.Massiv (
  -- * Central data-types
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
  KnownActivation,
  sActivation,
  withKnownActivation,
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
  logLikelihood,
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
  KnownLayerKind,
  withKnownLayerKind,
  sLayerKind,
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
  KnownNetwork,
  networkShape,
  withNetworkShape,
  NetworkShape (..),
  foldMapNetwork,
  foldZipNetwork,
) where

import Control.Applicative (liftA2)
import Control.Lens (Lens', lens, _1, _2)
import Control.Subcategory.Linear
import qualified Data.Bifunctor as Bi
import Data.Generics.Labels ()
import Data.List (iterate')
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array.Manifest.Vector as VM
import Data.Proxy (Proxy (..))
import Data.Strict (Pair (..))
import qualified Data.Strict.Tuple as SP
import Data.Tuple (swap)
import Data.Type.Natural
import qualified Data.Vector.Generic as G
import qualified Data.Vector.Unboxed as U
import DeepLearning.NeuralNetowrk.Massiv.Types
import Generic.Data
import Numeric.Backprop
import Numeric.Function.Activation (relu, sigmoid, softmax)
import Numeric.Natural (Natural)
import System.Random.MWC.Distributions (normal)
import System.Random.Stateful

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
    $ const
    $ \BatchLayer {..} -> BatW {..}

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
                batchMean = sumRows' x /. m
                xRel = x - duplicateAsCols' batchMean
                batchDev = sumRows' (xRel * xRel) /. m
                xHat = xRel / duplicateAsCols' (sqrt (batchDev +. 1e-12))
                x' = duplicateAsCols' gamma * xHat + duplicateAsCols' beta
                bRP =
                  isoVar2 BatRP (\(BatRP z s) -> (z, s)) batchMean batchDev
             in T2 x' bRP
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

data TopLayerView h i l k xs o a = TopLayerView
  { topLayer :: !(h l i k a)
  , continue :: !(Network h k xs o a)
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

data ResidualView h i fs xs o a = ResidualView
  { block :: !(Network h i fs i a)
  , continue :: !(Network h i xs o a)
  }
  deriving (Generic)

deriving anyclass instance
  ( U.Unbox a
  , Num a
  , KnownNat i
  , KnownNat o
  , forall l' x y. (KnownNat x, KnownNat y) => Backprop (h l' x y a)
  ) =>
  Backprop (ResidualView h i fs xs o a)

residualL ::
  (KnownNetwork i fs i) =>
  Lens' (Network h i (Skip fs ': xs) o a) (ResidualView h i fs xs o a)
residualL =
  lens
    ( \(hlika ::- net) ->
        ResidualView {block = hlika, continue = net}
    )
    (\_ ResidualView {..} -> block ::- continue)

runNN ::
  forall m i hs o a s.
  ( KnownNat m
  , U.Unbox a
  , RealFloat a
  , Backprop a
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
      Network RecParams h ls o a ->
      BVar s (Network Weights h ls o a) ->
      BVar s (UMat m h a) ->
      BVar s (UMat m o a, Network RecParams h ls o a)
    {-# INLINE go #-}
    go Output _ = flip T2 (auto Output)
    -- Single layer
    go (ps :- restPs) lay = \x ->
      let !decons = lay ^^. topLayerL
          !top = decons ^^. #topLayer
          !vLays' = runLayer pass ps top x
          !passed = go restPs (decons ^^. #continue) (vLays' ^^. _1)
          !v' = passed ^^. _1
          !rest' = passed ^^. _2
       in T2 v' (isoVar2 (:-) (\(a :- b) -> (a, b)) (vLays' ^^. _2) rest')
    -- Residual connection
    go (blk ::- restPs) lay = \x ->
      let !resid = lay ^^. residualL
          !rep = resid ^^. #block
          !ran = runNN pass blk rep x
          !passed = go restPs (resid ^^. #continue) (ran ^^. _1 + x)
       in T2 (passed ^^. _1) $
            isoVar2 (::-) (\(a ::- b) -> (a, b)) (ran ^^. _2) (passed ^^. _2)

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
  Network RecParams i ls o a ->
  Network Weights i ls o a ->
  (Network Weights i ls o a, Network RecParams i ls o a)
gradNN loss (inps, oups) recPs =
  Bi.bimap ($ (1.0, 0.0)) snd
    . swap
    . backpropWith
      ( \net ->
          let !ysRecs' = runNN Train recPs net (auto inps)
           in T2 (loss (ysRecs' ^^. _1) (auto oups)) (ysRecs' ^^. _2)
      )

logLikelihood ::
  forall o m a.
  ( KnownNat o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , Floating a
  ) =>
  LossFunction m o a
logLikelihood yPred yTruth =
  negate $
    sumS (yTruth * log yPred + (1 - yTruth) * log (1 - yPred))

crossEntropy ::
  forall o m a.
  ( KnownNat o
  , KnownNat m
  , U.Unbox a
  , Backprop a
  , Floating a
  ) =>
  LossFunction m o a
crossEntropy yPred yTruth =
  negate $ sumS $ yTruth * log yPred

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
  withDataPair dataSet $ trainGD gamma alpha n loss

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
  forall m a i hs o k.
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
      let (dW, ps') = gradNN (\x y -> loss x y /. fromIntegral (dimVal @m)) dataSet ps ws
       in NeuralNetwork
            (alpha .* ps' + (1 - alpha) .* ps)
            (ws - gamma .* dW)

data AdamParams a = AdamParams {beta1, beta2, epsilon :: !a}
  deriving (Show, Eq, Ord, Generic, Functor, Foldable, Traversable)

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
  withDataPair dataSet $ trainAdam gamma alpha ap n loss

trainAdam ::
  forall m i hs o a k.
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
  , Backprop k
  , VectorSpace k a
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
{-# SPECIALIZE trainAdam_ ::
  ( KnownNetwork i hs o
  , KnownNat m
  ) =>
  -- \| Learning Rate (dt)
  Double ->
  -- \| Dumping factor
  Double ->
  AdamParams Double ->
  Int ->
  LossFunction m o Double ->
  (UMat m i Double, UMat m o Double) ->
  NeuralNetwork i hs o Double ->
  NeuralNetwork i hs o Double
  #-}
trainAdam_ gamma alpha AdamParams {..} n loss dataSet =
  SP.fst . last . take n . iterate' step . (:!: (reps 0.0 :!: reps 0.0))
  where
    step ((NeuralNetwork ps net) :!: (s :!: v)) =
      let (dW, ps') =
            gradNN (\x y -> loss x y /. fromIntegral (dimVal @m)) dataSet ps net
          sN = beta2 .* s + (1.0 - beta2) .* (dW * dW)
          vN = beta1 .* v + (1.0 - beta1) .* dW
          !net' = net - (gamma .* vN) / sqrt (sN +. epsilon)
          !ps'' = alpha .* ps' + (1 - alpha) .* ps
       in NeuralNetwork ps'' net' :!: (sN :!: vN)

randomNetwork ::
  ( RandomGenM g r m
  , KnownNat i
  , KnownNat o
  , U.Unbox a
  , Floating a
  , Real a
  ) =>
  g ->
  Network LayerSpec i ls o a ->
  m (NeuralNetwork i ls o a)
randomNetwork g =
  liftA2 NeuralNetwork
    <$> hmapNetworkM' \case
      (AffP _ :: LayerSpec _ i _ _) -> pure AffRP
      (LinP _ :: LayerSpec _ i _ _) -> pure LinRP
      (ActP :: LayerSpec _ _ _ _) -> pure $ ActRP sActivation
      (BNP :: LayerSpec _ i _ _) -> pure $ BatRP 0.0 1.0
    <*> hmapNetworkM' \case
      (AffP s :: LayerSpec _ i _ _) -> do
        ws <- replicateMatA $ realToFrac <$> normal 0.0 (realToFrac s) g
        pure $ AffW ws 0.0
      (LinP s :: LayerSpec _ i _ _) -> do
        ws <- replicateMatA $ realToFrac <$> normal 0.0 (realToFrac s) g
        pure $ LinW ws
      (ActP :: LayerSpec _ _ _ _) -> pure ActW
      (BNP :: LayerSpec _ i _ _) -> pure $ BatW 1.0 0.0

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
