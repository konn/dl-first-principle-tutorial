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
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PatternSynonyms #-}
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
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module DeepLearning.NeuralNetowrk.HigherKinded
  ( Layer (..),
    Activation (..),
    ActivatorProxy,
    reLUP,
    sigmoidP,
    tanhP,
    idP,
    Activation' (..),
    HProxy (..),
    reLUA,
    sigmoidA,
    tanhA,
    idA,
    Network (..),
    NeuralNetwork,
    GradientStack,
    Gradients (.., Grads),
    Weights (..),
    WeightStack,
    LossFunction,
    evalLayer,
    evalNN,
    applyActivator,
    trainGD,
    toGradientStack,
    mapNetwork,
    zipNetworkWith,
    zipNetworkWith3,
    htraverseNetwork,
    crossEntropy,
    generateNetworkA,
    AdamParams (..),
    trainAdam,
    randomNetwork,
  )
where

import qualified Control.Foldl as L
import Control.Lens (alaf, foldMapOf)
import Control.Monad (join)
import Data.Functor.Compose (Compose (..))
import Data.Kind (Constraint)
import Data.List (iterate')
import Data.Monoid (Sum (..))
import Data.Reflection (Reifies)
import Data.Strict (Pair (..))
import qualified Data.Strict as SP
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import Generic.Data
import Linear
import Numeric.AD (auto, grad)
import Numeric.AD.Internal.Reverse (Tape)
import Numeric.AD.Mode.Reverse (Reverse)
import Numeric.Function.Activation (relu, sigmoid)
import System.Random.MWC.Distributions (normal, standard)
import System.Random.Stateful (RandomGenM)

data Activation = ReLU | Sigmoid | Tanh | Id
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

type ActivatorProxy = HProxy Activation

sigmoidP, reLUP, tanhP, idP :: forall o i a. ActivatorProxy i o a
sigmoidP = HProxy Sigmoid
reLUP = HProxy ReLU
tanhP = HProxy Tanh
idP = HProxy Id

newtype HProxy p i o a = HProxy {runHProxy :: p}
  deriving (Show, Eq, Ord, Generic, Functor, Foldable, Traversable, Generic1)
  deriving (Applicative) via Generically1 (HProxy p i o)

data Activation' m i o a = Activation' {getActivation :: Activation, seed :: m a}
  deriving (Show, Eq, Ord, Generic, Bounded, Functor, Foldable, Traversable)

data Layer i o a = Layer' (Weights i o a) Activation
  deriving (Show, Eq, Ord, Generic1, Generic, Functor, Foldable, Traversable)

pattern Layer :: o (i a) -> o a -> Activation -> Layer i o a
pattern Layer w b a = Layer' (Weights w b) a

{-# COMPLETE Layer #-}

reLUA, sigmoidA, idA, tanhA :: forall i o f a. f a -> Activation' f i o a
reLUA = Activation' ReLU
sigmoidA = Activation' Sigmoid
idA = Activation' Id
tanhA = Activation' Tanh

newtype Gradients i o a = Grads' (Weights i o a)
  deriving (Show, Eq, Ord, Generic1, Generic, Functor, Foldable, Traversable)

pattern Grads :: o (i a) -> o a -> Gradients i o a
pattern Grads dW dB = Grads' (Weights dW dB)

{-# COMPLETE Grads #-}

deriving via
  Generically1 (Gradients i o)
  instance
    (Applicative i, Applicative o) => Applicative (Gradients i o)

data Weights i o a = Weights !(o (i a)) !(o a)
  deriving (Show, Eq, Ord, Generic1, Generic, Functor, Foldable, Traversable)
  deriving anyclass (Additive)

type WeightStack = Network Weights

deriving via
  Generically1 (Weights i o)
  instance
    (Applicative i, Applicative o) => Applicative (Weights i o)

applyActivator :: RealFloat a => Activation -> a -> a
applyActivator ReLU = relu
applyActivator Sigmoid = sigmoid
applyActivator Tanh = tanh
applyActivator Id = id

evalLayer ::
  ( RealFloat a
  , Foldable i
  , Additive o
  , Additive i
  ) =>
  Layer i o a ->
  i a ->
  o a
evalLayer (Layer w b f) x =
  applyActivator f <$> (w !* x ^+^ b)

-- Can we existentialise internal layers?
data Network h i fs o a where
  Output :: Network h i '[] i a
  (:-) ::
    -- TODO: Drop these ad-hoc constraints and move to 'All'
    (Traversable k, Metric k, Applicative k) =>
    !(h i k a) ->
    !(Network h k fs o a) ->
    Network h i (k ': fs) o a

instance Show (Network h i '[] i a) where
  showsPrec _ Output = showString "Output"
  {-# INLINE showsPrec #-}

instance
  (Show (h i k a), Show (Network h k ks o a)) =>
  Show (Network h i (k ': ks) o a)
  where
  showsPrec d (a :- as) =
    showParen (d > 9) $
      showsPrec 10 a . showString " :- " . showsPrec 9 as

type NeuralNetwork = Network Layer

infixr 9 :-

type GradientStack = Network Gradients

type Result :: x -> [x] -> x
type family Result f fs where
  Result f '[] = f
  Result _ (x ': xs) = Result x xs

type All :: (k -> Constraint) -> [k] -> Constraint
type family All c xs where
  All c '[] = ()
  All c (x ': xs) = (c x, All c xs)

evalNN ::
  (RealFloat a, Foldable i, Additive i) =>
  NeuralNetwork i ls o a ->
  i a ->
  o a
evalNN Output !xs = xs
evalNN (l :- net') !xs = evalNN net' $! evalLayer l xs

instance
  (Functor i, forall x y. (Functor x, Functor y) => Functor (h x y)) =>
  Functor (Network h i ls o)
  where
  fmap _ Output = Output
  fmap f (hfka :- net') = fmap f hfka :- fmap f net'

instance
  (Foldable i, forall x y. (Foldable x, Foldable y) => Foldable (h x y)) =>
  Foldable (Network h i ls o)
  where
  foldMap _ Output = mempty
  foldMap f (hfka :- net') = foldMap f hfka <> foldMap f net'
  {-# INLINE foldMap #-}

instance
  {-# OVERLAPPING #-}
  (Traversable i) =>
  Traversable (Network Layer i ls o)
  where
  traverse _ Output = pure Output
  traverse f (hfka :- net') =
    (:-) <$> traverse f hfka <*> traverse f net'
  {-# INLINE traverse #-}

instance
  {-# OVERLAPPING #-}
  (Traversable i) =>
  Traversable (Network Gradients i ls o)
  where
  traverse _ Output = pure Output
  traverse f (hfka :- net') =
    (:-) <$> traverse f hfka <*> traverse f net'
  {-# INLINE traverse #-}

mapNetwork ::
  (Traversable i, Applicative i, Metric i) =>
  ( forall x y.
    ( Traversable x
    , Metric x
    , Applicative x
    , Metric y
    , Traversable y
    , Applicative y
    ) =>
    h x y a ->
    k x y b
  ) ->
  Network h i ls o a ->
  Network k i ls o b
{-# INLINE mapNetwork #-}
mapNetwork _ Output = Output
mapNetwork f (hfka :- net') = f hfka :- mapNetwork f net'

htraverseNetwork ::
  (Traversable i, Metric i, Applicative f) =>
  ( forall x y.
    (Traversable x, Metric x, Metric y, Traversable y) =>
    h x y a ->
    f (k x y b)
  ) ->
  Network h i ls o a ->
  f (Network k i ls o b)
{-# INLINE htraverseNetwork #-}
htraverseNetwork _ Output = pure Output
htraverseNetwork f (hfka :- net') =
  (:-) <$> f hfka <*> htraverseNetwork f net'

zipNetworkWith ::
  forall h k t i ls o a b c.
  (Traversable i, Metric i, Applicative i) =>
  ( forall x y.
    ( Traversable x
    , Applicative x
    , Metric x
    , Traversable y
    , Applicative y
    , Metric y
    ) =>
    h x y a ->
    k x y b ->
    t x y c
  ) ->
  Network h i ls o a ->
  Network k i ls o b ->
  Network t i ls o c
{-# INLINE zipNetworkWith #-}
zipNetworkWith f = go
  where
    go ::
      (Traversable f', Metric f', Applicative f') =>
      Network h f' ls' g' a ->
      Network k f' ls' g' b ->
      Network t f' ls' g' c
    go Output Output = Output
    go (hxy :- hs) (kxy :- ks) = f hxy kxy :- go hs ks

zipNetworkWith3 ::
  forall h k t u i ls o a b c d.
  (Traversable i, Metric i, Applicative i) =>
  ( forall x y.
    ( Traversable x
    , Applicative x
    , Metric x
    , Traversable y
    , Applicative y
    , Metric y
    ) =>
    h x y a ->
    k x y b ->
    t x y c ->
    u x y d
  ) ->
  Network h i ls o a ->
  Network k i ls o b ->
  Network t i ls o c ->
  Network u i ls o d
{-# INLINE zipNetworkWith3 #-}
zipNetworkWith3 f = go
  where
    go ::
      (Traversable f', Metric f', Applicative f') =>
      Network h f' ls' g' a ->
      Network k f' ls' g' b ->
      Network t f' ls' g' c ->
      Network u f' ls' g' d
    go Output Output Output = Output
    go (hxy :- hs) (kxy :- ks) (txy :- ts) = f hxy kxy txy :- go hs ks ts

toGradientStack ::
  (Traversable i, Applicative i, Metric i) =>
  NeuralNetwork i hs o a ->
  GradientStack i hs o a
toGradientStack = mapNetwork $ \(Layer k ka _) -> Grads k ka

type LossFunction o a =
  forall s.
  (Reifies s Tape) =>
  o (Reverse s a) ->
  o (Reverse s a) ->
  Reverse s a

pass ::
  forall ls i o a.
  ( RealFloat a
  , Metric i
  , Traversable i
  , Applicative i
  , Functor o
  , U.Unbox (i a)
  , U.Unbox (o a)
  ) =>
  -- | Loss function
  LossFunction o a ->
  U.Vector (i a, o a) ->
  NeuralNetwork i ls o a ->
  GradientStack i ls o a
pass loss dataSet =
  toGradientStack
    . grad
      ( \net ->
          alaf
            Sum
            (foldMapOf vectorTraverse)
            (\(xs0, ys0) -> loss (evalNN net $ auto <$> xs0) $ auto <$> ys0)
            dataSet
      )

trainGD ::
  (U.Unbox (i a), U.Unbox (o a), Applicative i, Metric i, Traversable i, Functor o) =>
  RealFloat a =>
  a ->
  Int ->
  LossFunction o a ->
  U.Vector (i a, o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainGD gamma n loss dataSet = last . take n . iterate' step
  where
    step net =
      zipNetworkWith
        (\(Layer' w act) (Grads' dW) -> Layer' (w ^-^ gamma *^ dW) act)
        net
        (pass loss dataSet net)

data AdamParams a = AdamParams {beta1, beta2, epsilon :: !a}
  deriving (Show, Eq, Ord, Generic)

trainAdam ::
  forall i hs o a.
  ( U.Unbox (i a)
  , U.Unbox (o a)
  , Applicative i
  , Metric i
  , Traversable i
  , Functor o
  , Applicative (WeightStack i hs o)
  ) =>
  RealFloat a =>
  -- | Learning Rate
  a ->
  AdamParams a ->
  Int ->
  LossFunction o a ->
  U.Vector (i a, o a) ->
  NeuralNetwork i hs o a ->
  NeuralNetwork i hs o a
trainAdam gamma AdamParams {..} n loss dataSet =
  SP.fst . last . take n . iterate' step . (:!: (s0 :!: v0))
  where
    v0, s0 :: WeightStack i hs o a
    !v0@s0 = pure 0.0
    step (net :!: (s :!: v)) =
      let dW = pass loss dataSet net
          sN = zipNetworkWith f2 s dW
          vN = zipNetworkWith f3 v dW
       in zipNetworkWith3 f net vN sN :!: (sN :!: vN)

    f ::
      (Additive f, Applicative f, Additive g, Applicative g) =>
      Layer f g a ->
      Weights f g a ->
      Weights f g a ->
      Layer f g a
    f (Layer' ws sf) vs ss =
      Layer' (ws ^-^ ((/) <$> gamma *^ vs <*> fmap (sqrt . (epsilon +)) ss)) sf

    f2 ::
      (Additive f, Applicative f, Additive g, Applicative g) =>
      Weights f g a ->
      Gradients f g a ->
      Weights f g a
    f2 sW (Grads' dW) =
      beta2 *^ sW ^+^ (1 - beta2) *^ (join (*) <$> dW)

    f3 ::
      (Additive f, Applicative f, Additive g, Applicative g) =>
      Weights f g a ->
      Gradients f g a ->
      Weights f g a
    f3 vW (Grads' dW) =
      beta1 *^ vW ^+^ (1 - beta1) *^ dW

crossEntropy :: (Foldable f, Applicative f, Floating a) => LossFunction f a
crossEntropy ys' ys =
  L.fold (-L.mean) $ l <$> ys' <*> ys
  where
    l y' y = y * log y' + (1 - y) * log (1 - y')

instance
  ( forall x y. (Functor x, Functor y) => Functor (h x y)
  , Functor f
  , g ~ f
  ) =>
  Applicative (Network h f '[] g)
  where
  pure = const Output
  {-# INLINE pure #-}
  (<*>) = const $ const Output
  {-# INLINE (<*>) #-}

instance
  ( forall x y. (Functor x, Functor y) => Functor (h x y)
  , Applicative (Network h k ls o)
  , Applicative (h i k)
  , Traversable k
  , Metric k
  , Applicative k
  , Functor i
  ) =>
  Applicative (Network h i (k ': ls) o)
  where
  pure x = pure x :- pure x
  {-# INLINE pure #-}
  (f :- fs) <*> (x :- xs) = (f <*> x) :- (fs <*> xs)
  {-# INLINE (<*>) #-}

generateNetworkA ::
  (Traversable i, Applicative f, Applicative i, Metric i) =>
  Network (Activation' f) i ls o a ->
  f (Network Layer i ls o a)
generateNetworkA =
  sequenceA . mapNetwork (\(Activation' act val) -> Layer' (pure val) act)

randomNetwork ::
  forall i ls o g r m.
  ( RandomGenM g r m
  , Traversable i
  , Applicative i
  ) =>
  g ->
  Network ActivatorProxy i ls o Double ->
  m (NeuralNetwork i ls o Double)
randomNetwork _ Output = pure Output
randomNetwork g (HProxy act :- net') = do
  let s =
        case act of
          ReLU -> sqrt $ recip $ fromIntegral (length $ pure @i ()) / 2.0
          _ -> sqrt $ recip $ fromIntegral $ length $ pure @i ()

  Compose ws <- sequence $ pure $ normal 0.0 s g
  bs <- sequence $ pure $ standard g
  (Layer ws bs act :-) <$> randomNetwork g net'
