{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
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

{-# HLINT ignore "Use id" #-}
{-# HLINT ignore "Eta reduce" #-}

module DeepLearning.NeuralNetowrk.Massiv.Types (
  -- * Central data-types
  NeuralNetwork (..),
  SomeNeuralNetwork (..),
  SomeNetwork (..),
  activationVal,
  Network (..),
  RecParams (..),
  Weights (..),
  KnownActivation,
  Activation (..),
  SActivation (..),
  sActivation,
  SomeActivation (..),
  someActivation,
  type L,
  type Skip,
  LayerKind (..),
  NetworkStat (..),
  networkStat,
  LayerInfo (..),

  -- ** Network construction
  LayerSpec (..),

  -- * General execution with backpropagation
  SLayerKind (..),
  KnownLayerKind,
  sLayerKind,
  withKnownLayerKind,
  LayerLike,
  Aff,
  Lin,
  Act,
  BN,
  LN,

  -- * Operators for manipulating networks
  withKnownNeuralNetwork,
  mapNetwork,
  htraverseNetwork,
  hmapNetworkM',
  zipNetworkWith,
  zipNetworkWith3,
  KnownNetwork,
  networkShape,
  withNetworkShape,
  NetworkShape (..),
  foldMapNetwork,
  foldZipNetwork,
  withKnownActivation,
) where

import Control.Arrow ((>>>))
import Control.DeepSeq (NFData (..))
import Control.Monad (when)
import Control.Subcategory.Linear
import qualified Data.Bifunctor as Bi
import qualified Data.DList as DL
import Data.Generics.Labels ()
import Data.Kind (Type)
import Data.Monoid (All (..), Ap (..), Sum (..))
import Data.Persist (Get, Persist (..), Put)
import Data.Proxy (Proxy (..))
import Data.Reflection (Given (..), give)
import Data.Type.Equality (testEquality, type (:~:) (..))
import Data.Type.Natural
import qualified Data.Vector.Unboxed as U
import Generic.Data
import Numeric.Backprop
import Type.Reflection (typeRep)

data LayerKind
  = -- | Affine transformation
    Aff
  | -- | Linear transformation layer without bias (to be combined with layer or batch nomarlisation)
    Lin
  | -- | Activation layer
    Act Activation
  | -- | Batch normalisation layer
    BN
  | -- | Layer normalisation layer
    LN
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

deriving instance Eq (SActivation a)

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

type KnownActivation a = Given (SActivation a)

withKnownActivation :: SActivation a -> (KnownActivation a => r) -> r
withKnownActivation = give

sActivation :: KnownActivation a => SActivation a
sActivation = given

instance Given (SActivation 'ReLU) where
  given = SReLU
  {-# INLINE given #-}

instance Given (SActivation 'Sigmoid) where
  given = SSigmoid
  {-# INLINE given #-}

instance Given (SActivation 'Tanh) where
  given = STanh
  {-# INLINE given #-}

instance Given (SActivation 'Softmax) where
  given = SSoftmax
  {-# INLINE given #-}

instance Given (SActivation 'Id) where
  given = SId
  {-# INLINE given #-}

data SLayerKind (l :: LayerKind) n m where
  SAff :: SLayerKind 'Aff n m
  SLin :: SLayerKind 'Lin n m
  SAct :: SActivation a -> SLayerKind ( 'Act a) n n
  SBN :: SLayerKind 'BN n n
  SLN :: SLayerKind 'LN n n

withKnownLayerKind :: (KnownNat n, KnownNat m) => SLayerKind l n m -> (KnownLayerKind l n m => r) -> r
{-# INLINE withKnownLayerKind #-}
withKnownLayerKind s f = give s f

type KnownLayerKind l n m = (KnownNat n, KnownNat m, Given (SLayerKind l n m))

sLayerKind :: forall l n m. KnownLayerKind l n m => SLayerKind l n m
sLayerKind = given

instance Given (SLayerKind 'Aff n m) where
  given = SAff
  {-# INLINE given #-}

instance Given (SLayerKind 'Lin n m) where
  given = SLin
  {-# INLINE given #-}

instance (n ~ m, KnownActivation a) => Given (SLayerKind ( 'Act a) n m) where
  given = SAct sActivation
  {-# INLINE given #-}

instance (n ~ m) => Given (SLayerKind 'BN n m) where
  given = SBN
  {-# INLINE given #-}

instance (n ~ m) => Given (SLayerKind 'LN n m) where
  given = SLN
  {-# INLINE given #-}

type Aff = 'Aff

type Lin = 'Lin

type Act = 'Act

type BN = 'BN

type LN = 'LN

type LayerLike = LayerKind -> Nat -> Nat -> Type -> Type

type LayerSpec :: LayerLike
data LayerSpec l n m a where
  AffP :: a -> LayerSpec 'Aff n m a
  LinP :: a -> LayerSpec 'Lin n m a
  ActP :: KnownActivation act => LayerSpec ( 'Act act) n n a
  BNP :: LayerSpec 'BN n n a
  LNP :: LayerSpec 'LN n n a

deriving instance Show a => Show (LayerSpec l n m a)

deriving instance Eq a => Eq (LayerSpec l n m a)

deriving instance Ord a => Ord (LayerSpec l n m a)

-- | Weights optimised by backprop.
type Weights :: LayerKind -> Nat -> Nat -> Type -> Type
data Weights l i o a where
  AffW :: !(UMat i o a) -> !(UVec o a) -> Weights 'Aff i o a
  LinW :: !(UMat i o a) -> Weights 'Lin i o a
  ActW :: Weights ( 'Act act) i i a
  BatW :: {scale, shift :: !(UVec i a)} -> Weights 'BN i i a
  LayerNormW :: {scaleLN, shiftLN :: !(UVec i a)} -> Weights 'LN i i a

deriving instance (Eq a, U.Unbox a) => Eq (Weights l i o a)

deriving instance (Show a, U.Unbox a) => Show (Weights l i o a)

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
liftBinWs _ fV (LayerNormW w b) (LayerNormW w' b') = LayerNormW (fV w w') (fV b b')

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
liftUnWs _ fV (LayerNormW scale shift) = LayerNormW (fV scale) (fV shift)

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
    SLN -> LayerNormW <$> reps <*> reps
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
  LayerNormW w b >.< LayerNormW w' b' = w >.< w' + b >.< b'
  {-# INLINE (>.<) #-}
  sumS (AffW w b) = sumS w + sumS b
  sumS (LinW w) = sumS w
  sumS ActW = 0
  sumS (BatW mu sigma) = sumS mu + sumS sigma
  sumS (LayerNormW mu sigma) = sumS mu + sumS sigma
  {-# INLINE sumS #-}

-- | Parameters updated by step-by-step, but not optimised by backprop.
type RecParams :: LayerKind -> Nat -> Nat -> Type -> Type
data RecParams l i o a where
  AffRP :: RecParams 'Aff i o a
  LinRP :: RecParams 'Lin i o a
  ActRP :: !(SActivation act) -> RecParams ( 'Act act) i i a
  BatRP :: {mean, deviation :: !(UVec i a)} -> RecParams 'BN i i a
  LayerNormRP :: RecParams 'LN i i a

deriving instance (Eq a, U.Unbox a) => Eq (RecParams l i o a)

deriving instance (Show a, U.Unbox a) => Show (RecParams l i o a)

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
liftBinRP _ l@LayerNormRP LayerNormRP {} = l

liftUnRP ::
  (UVec o a -> UVec o a) ->
  RecParams l i o a ->
  RecParams l i o a
{-# INLINE liftUnRP #-}
liftUnRP _ l@AffRP = l
liftUnRP _ l@LinRP = l
liftUnRP _ l@(ActRP _) = l
liftUnRP fV (BatRP mu sigma) = BatRP (fV mu) (fV sigma)
liftUnRP _ l@LayerNormRP = l

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
    SLN -> const LayerNormRP
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
  LayerNormRP >.< LayerNormRP = 0
  {-# INLINE (>.<) #-}
  sumS AffRP = 0
  sumS LinRP = 0
  sumS (ActRP _) = 0
  sumS (BatRP mu sigma) = sumS mu + sumS sigma
  sumS LayerNormRP = 0
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
    l@LayerNormRP -> l
    BatRP l r -> BatRP (zero l) (zero r)
  one = \case
    AffRP -> AffRP
    LinRP -> LinRP
    l@LayerNormRP -> l
    ActRP s -> ActRP s
    BatRP l r -> BatRP (one l) (one r)
  add l@AffRP AffRP {} = l
  add l@LinRP LinRP {} = l
  add l@(ActRP _) ActRP {} = l
  add (BatRP mu sigma) (BatRP mu' sigma') =
    BatRP (add mu mu') (add sigma sigma')
  add l@LayerNormRP LayerNormRP {} = l
  {-# INLINE add #-}

instance (forall l x y. Semigroup (h l x y a)) => Semigroup (Network h i ls o a) where
  Output <> Output = Output
  (a :- as) <> (b :- bs) = (a <> b) :- (as <> bs)
  (blk ::- as) <> (blk' ::- bs) = (blk <> blk') ::- (as <> bs)
  {-# INLINE (<>) #-}

-- | FIXME: defining on the case of @l@ would reduce runtime branching.
instance (KnownNat i, KnownNat o, U.Unbox a, Num a) => Backprop (Weights l i o a) where
  zero = \case
    AffW {} -> AffW 0 0
    LinW {} -> LinW 0
    l@ActW {} -> l
    BatW {} -> BatW 0 0
    LayerNormW {} -> LayerNormW 0 0
  {-# INLINE zero #-}
  one = \case
    AffW {} -> AffW 1 1
    LinW {} -> LinW 1
    l@ActW {} -> l
    BatW {} -> BatW 1 1
    LayerNormW {} -> LayerNormW 1 1
  {-# INLINE one #-}
  add (AffW mat vec) (AffW mat' vec') = AffW (add mat mat') (add vec vec')
  add (LinW mat) (LinW mat') = LinW (add mat mat')
  add l@ActW {} ActW {} = l
  add (BatW a b) (BatW a' b') = BatW (a + a') (b + b')
  add (LayerNormW a b) (LayerNormW a' b') = LayerNormW (a + a') (b + b')
  {-# INLINE add #-}

data Spec = L LayerKind Nat | Skip [Spec]
  deriving (Generic)

type L = 'L

type Skip = 'Skip

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
  (::-) ::
    (KnownNat i, KnownNetwork i hs i) =>
    !(Network h i hs i a) ->
    !(Network h i fs o a) ->
    Network h i (Skip hs ': fs) o a

infixr 9 :-, ::-

data NeuralNetwork i ls o a = NeuralNetwork
  { recParams :: !(Network RecParams i ls o a)
  , weights :: !(Network Weights i ls o a)
  }
  deriving (Generic, Eq)
  deriving anyclass (NFData)

instance
  (KnownNetwork i ls o, Persist a, U.Unbox a) =>
  Persist (NeuralNetwork i ls o a)
  where
  put NeuralNetwork {..} = do
    put $ toNetworkHeader $ networkShape @i @ls @o
    putNetworkBody recParams
    putNetworkBody weights
  get = do
    hdr <- get
    let sh = networkShape @i @ls @o
    when (hdr /= toNetworkHeader sh) $
      fail $
        "NetworkHeader mismatched; (expected, got) = "
          <> show (toNetworkHeader sh, hdr)
    recParams <- getNetworkBodyWith sh
    weights <- getNetworkBodyWith sh
    pure NeuralNetwork {..}

data SomeNeuralNetwork i o a where
  MkSomeNeuralNetwork :: NeuralNetwork i ls o a -> SomeNeuralNetwork i o a

instance Persist SomeNetworkShape where
  put (MkSomeNetworkShape sh) = put $ toNetworkHeader sh
  {-# INLINE put #-}
  get =
    either (fail . ("SomeNetworkShape: invalid network header: " <>)) pure
      . fromNetworkHeader
      =<< get

instance
  ( KnownNat i
  , KnownNat o
  , Persist a
  , U.Unbox a
  ) =>
  Persist (SomeNeuralNetwork i o a)
  where
  put (MkSomeNeuralNetwork (NeuralNetwork {..} :: NeuralNetwork i xs o a)) =
    withKnownNetwork recParams $ do
      put $ toNetworkHeader $ networkShape @i @xs @o
      putNetworkBody recParams
      putNetworkBody weights
  get = do
    MkSomeNetworkShape (sh :: NetworkShape i' xs o') <- get
    Refl <-
      maybe
        ( fail $
            "get/SomeNeuralNetwork: input mismatched (expected, got) = "
              <> show (dimVal @i, dimVal @i')
        )
        pure
        $ testEquality (typeRep @i) (typeRep @i')
    Refl <-
      maybe
        ( fail $
            "get/SomeNeuralNetwork: output mismatched (expected, got) = "
              <> show (dimVal @o, dimVal @o')
        )
        pure
        $ testEquality (typeRep @o) (typeRep @o')
    withNetworkShape sh $
      fmap MkSomeNeuralNetwork . NeuralNetwork <$> getNetworkBodyWith sh <*> getNetworkBodyWith sh

instance
  ( KnownNetwork i fs o
  , forall l x y. KnownLayerKind l x y => Num (h l x y a)
  ) =>
  Num (Network h i fs o a)
  where
  fromInteger = go $ networkShape @i @fs @o
    where
      go :: NetworkShape i' hs o' -> Integer -> Network h i' hs o' a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons _ rest) = (:-) <$> fromInteger <*> go rest
      go (IsSkip sh rest) = withNetworkShape sh $ (::-) <$> go sh <*> go rest
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
  ( Num a
  , KnownNetwork i fs o
  , Floating a
  , forall l x y. (KnownLayerKind l x y, KnownNat x, KnownNat y) => VectorSpace a (h l x y a)
  ) =>
  VectorSpace a (Network h i fs o a)
  where
  reps = go $ networkShape @i @fs @o
    where
      go :: NetworkShape i' hs o' -> a -> Network h i' hs o' a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons _ rest) = (:-) <$> reps <*> go rest
      go (IsSkip sh rest) = withNetworkShape sh $ (::-) <$> go sh <*> go rest
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
  ( KnownNetwork i fs o
  , forall l x y. (KnownLayerKind l x y => Fractional (h l x y a))
  ) =>
  Fractional (Network h i fs o a)
  where
  fromRational = go $ networkShape @i @fs @o
    where
      go :: NetworkShape i' hs o' -> Rational -> Network h i' hs o' a
      {-# INLINE go #-}
      go IsOutput = pure Output
      go (IsCons _ rest) = (:-) <$> fromRational <*> go rest
      go (IsSkip sh rest) = withNetworkShape sh $ (::-) <$> go sh <*> go rest
  (/) = zipNetworkWith (/)
  {-# INLINE (/) #-}
  recip = mapNetwork recip
  {-# INLINE recip #-}

instance
  ( U.Unbox a
  , Floating a
  , KnownNetwork i fs o
  , forall l x y. (KnownLayerKind l x y => Floating (h l x y a))
  ) =>
  Floating (Network h i fs o a)
  where
  pi = go $ networkShape @i @fs @o
    where
      go :: NetworkShape i' hs o' -> Network h i' hs o' a
      {-# INLINE go #-}
      go IsOutput = Output
      go (IsCons _ rest) = pi :- go rest
      go (IsSkip sh rest) = withNetworkShape sh $ go sh ::- go rest
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
  ( forall l f g. (KnownNat f, KnownNat g, U.Unbox a) => Eq (h l f g a)
  , KnownNat i
  , KnownNat o
  , U.Unbox a
  ) =>
  Eq (Network h i fs o a)
  where
  (==) = fmap getAll . foldZipNetwork (fmap All . (==))
  {-# INLINE (==) #-}

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
  showsPrec d (residual ::- net') =
    showParen (d > 10) $
      showString "Residual "
        . showsPrec 10 residual
        . showString " ::- "
        . showsPrec 9 net'

data NetworkShape i xs o where
  IsOutput :: NetworkShape i '[] i
  IsCons :: (KnownLayerKind l i k, KnownNat k) => SLayerKind l i k -> NetworkShape k hs o -> NetworkShape i (L l k ': hs) o
  IsSkip ::
    KnownNat i =>
    NetworkShape i fs i ->
    NetworkShape i hs o ->
    NetworkShape i (Skip fs ': hs) o

data SomeNetworkShape where
  MkSomeNetworkShape ::
    (KnownNat i, KnownNat o) =>
    NetworkShape i xs o ->
    SomeNetworkShape

data SomeNetworkShape' i o where
  MkSomeNetworkShape' ::
    NetworkShape i xs o ->
    SomeNetworkShape' i o

data Block
  = Single !LayerKind !Word
  | Residual ![Block]
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (Persist)

data NetworkHeader = NetworkHeader
  { inputDim :: !Word
  , layerSpecs :: ![Block]
  , outputDim :: !Word
  }
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (Persist)

toNetworkHeader ::
  forall i xs o.
  (KnownNat i, KnownNat o) =>
  NetworkShape i xs o ->
  NetworkHeader
toNetworkHeader shape =
  NetworkHeader
    { inputDim = fromIntegral $ dimVal @i
    , layerSpecs = demoteLayerSpecs shape
    , outputDim = fromIntegral $ dimVal @o
    }

fromNetworkHeader :: NetworkHeader -> Either String SomeNetworkShape
fromNetworkHeader hdr@NetworkHeader {..} =
  case (someNatVal (fromIntegral inputDim), someNatVal (fromIntegral outputDim)) of
    (SomeNat (_ :: Proxy i), SomeNat (_ :: Proxy o)) -> do
      MkSomeNetworkShape' net <-
        Bi.first (("While decoding: " <> show hdr) <>) $
          fromNetworkHeader' @i @o layerSpecs
      pure $ MkSomeNetworkShape net

fromNetworkHeader' ::
  forall i o.
  (KnownNat i, KnownNat o) =>
  [Block] ->
  Either String (SomeNetworkShape' i o)
fromNetworkHeader' [] = do
  Refl <- dimCheck @i @o "Terminal node"
  pure $ MkSomeNetworkShape' IsOutput
fromNetworkHeader' (Single kind dim : rest) =
  case someNatVal $ fromIntegral dim of
    SomeNat (_ :: Proxy o') -> case kind of
      Aff -> do
        MkSomeNetworkShape' net <- fromNetworkHeader' @o' @o rest
        pure $ MkSomeNetworkShape' $ IsCons (SAff @i @o') net
      Lin -> do
        MkSomeNetworkShape' net <- fromNetworkHeader' @o' @o rest
        pure $ MkSomeNetworkShape' $ IsCons (SLin @i @o') net
      (Act ac) -> case someActivation ac of
        MkSomeActivation sact -> do
          Refl <- dimCheck @i @o' ("Activation layer (" <> show ac <> ")")
          MkSomeNetworkShape' net <- fromNetworkHeader' @i @o rest
          pure $ MkSomeNetworkShape' $ IsCons (SAct sact) net
      BN -> do
        Refl <- dimCheck @i @o' "Batchnorm layer"
        MkSomeNetworkShape' net <- fromNetworkHeader' @i @o rest
        pure $ MkSomeNetworkShape' $ IsCons SBN net
      LN -> do
        Refl <- dimCheck @i @o' "Layer normalisation layer"
        MkSomeNetworkShape' net <- fromNetworkHeader' @i @o rest
        pure $ MkSomeNetworkShape' $ IsCons SLN net
fromNetworkHeader' (Residual block : rest) = do
  MkSomeNetworkShape' block' <- fromNetworkHeader' @i @i block
  MkSomeNetworkShape' rest' <- fromNetworkHeader' @i @o rest
  pure $ MkSomeNetworkShape' $ IsSkip block' rest'

dimCheck :: forall i i'. (KnownNat i, KnownNat i') => String -> Either String (i :~: i')
dimCheck lab =
  case testEquality (typeRep @i) (typeRep @i') of
    Just r -> pure r
    Nothing ->
      Left $
        lab
          <> ": dimension mismatched; (expected, got) = "
          <> show (dimVal @i, dimVal @i')

demoteLayerSpecs :: NetworkShape i xs o -> [Block]
demoteLayerSpecs IsOutput = []
demoteLayerSpecs (IsCons (slk :: SLayerKind l k m) ns') =
  Single (demoteLayerKind slk) (fromIntegral $ dimVal @m) : demoteLayerSpecs ns'
demoteLayerSpecs (IsSkip (blk :: NetworkShape i hs i) ns') =
  Residual (demoteLayerSpecs blk) : demoteLayerSpecs ns'

demoteLayerKind :: forall l i k. SLayerKind l i k -> LayerKind
demoteLayerKind SAff = Aff
demoteLayerKind SLin = Lin
demoteLayerKind (SAct sa) = Act $ activationVal sa
demoteLayerKind SBN = BN
demoteLayerKind SLN = LN

type KnownNetwork i xs o = (KnownNat i, KnownNat o, Given (NetworkShape i xs o))

networkShape :: forall i xs o. KnownNetwork i xs o => NetworkShape i xs o
networkShape = given

instance (i ~ o) => Given (NetworkShape i '[] o) where
  given = IsOutput
  {-# INLINE given #-}

instance
  ( KnownLayerKind l i k
  , KnownNetwork k xs o
  ) =>
  Given (NetworkShape i (L l k : xs) o)
  where
  given = IsCons sLayerKind networkShape
  {-# INLINE given #-}

instance
  (KnownNetwork i hs i, KnownNetwork i xs o) =>
  Given (NetworkShape i (Skip hs : xs) o)
  where
  given = IsSkip networkShape networkShape

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
  zero (bk ::- hs) = zero bk ::- zero hs
  one Output = Output
  one (h :- hs) = one h :- one hs
  one (bk ::- hs) = one bk ::- one hs
  add Output Output = Output
  add (h :- hs) (g :- gs) = add h g :- add hs gs
  add (h ::- hs) (g ::- gs) = add h g ::- add hs gs

withKnownNeuralNetwork ::
  KnownNat i =>
  ( KnownNetwork i hs o =>
    NeuralNetwork i hs o a ->
    r
  ) ->
  NeuralNetwork i hs o a ->
  r
withKnownNeuralNetwork f n = withKnownNetwork (recParams n) (f n)

withNetworkShape :: (KnownNat i, KnownNat o) => NetworkShape i hs o -> (KnownNetwork i hs o => r) -> r
withNetworkShape sh f = give sh f

withKnownNetwork ::
  KnownNat i =>
  Network h i hs o a ->
  ( KnownNetwork i hs o =>
    r
  ) ->
  r
withKnownNetwork Output f = f
withKnownNetwork (_ :- ps) f = withKnownNetwork ps f
withKnownNetwork (blk ::- ps) f = withKnownNetwork blk $ withKnownNetwork ps f

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
mapNetwork f (blk ::- net') = mapNetwork f blk ::- mapNetwork f net'

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
htraverseNetwork f (blk ::- net') =
  (::-) <$> htraverseNetwork f blk <*> htraverseNetwork f net'

hmapNetworkM' ::
  forall i o a h f k b ls.
  (Monad f) =>
  ( forall l x y.
    (KnownNat x, KnownNat y) =>
    h l x y a ->
    f (k l x y b)
  ) ->
  Network h i ls o a ->
  f (Network k i ls o b)
{-# INLINE hmapNetworkM' #-}
hmapNetworkM' f = go
  where
    go :: Network h i' ls' o' a -> f (Network k i' ls' o' b)
    go Output = pure Output
    go (hfka :- net') = do
      !b <- f hfka
      !bs <- go net'
      pure $! b :- bs
    go (blk ::- net') = do
      !blk' <- go blk
      !bs <- go net'
      pure $! blk' ::- bs

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
foldMapNetwork f = go mempty
  where
    go :: w -> Network h x hs y a -> w
    {-# INLINE go #-}
    go !w Output = w
    go !w (h :- hs) = go (w <> f h) hs
    go !w (h ::- hs) = go (go w h) hs

foldZipNetwork ::
  forall h g i ls o a b w.
  (Monoid w) =>
  ( forall l x y.
    (KnownNat x, KnownNat y, KnownLayerKind l x y) =>
    h l x y a ->
    g l x y b ->
    w
  ) ->
  Network h i ls o a ->
  Network g i ls o b ->
  w
foldZipNetwork f = go mempty
  where
    go ::
      w ->
      Network h x hs y a ->
      Network g x hs y b ->
      w
    {-# INLINE go #-}
    go !w Output Output = w
    go !w (h :- hs) (g :- gs) = go (w <> f h g) hs gs
    go !w (h ::- hs) (g ::- gs) = go (go w h g) hs gs

zipNetworkWith ::
  forall h k t i ls o a b c.
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
      Network h n' ls' m' a ->
      Network k n' ls' m' b ->
      Network t n' ls' m' c
    go Output Output = Output
    go (hxy :- hs) (kxy :- ks) =
      let !c = f hxy kxy
          !rest = go hs ks
       in c :- rest
    go (hxy ::- hs) (kxy ::- ks) =
      let !c = go hxy kxy
          !rest = go hs ks
       in c ::- rest

zipNetworkWith3 ::
  forall h k t u i ls o a b c d.
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
      Network h n' ls' m' a ->
      Network k n' ls' m' b ->
      Network t n' ls' m' c ->
      Network u n' ls' m' d
    go Output Output Output = Output
    go (hxy :- hs) (kxy :- ks) (txy :- ts) =
      let !d = f hxy kxy txy in d :- go hs ks ts
    go (hxy ::- hs) (kxy ::- ks) (txy ::- ts) =
      let !d = go hxy kxy txy in d ::- go hs ks ts

data AdamParams a = AdamParams {beta1, beta2, epsilon :: !a}
  deriving (Show, Eq, Ord, Generic)

data SomeNetwork h i o a where
  MkSomeNetwork :: KnownNetwork i hs o => Network h i hs o a -> SomeNetwork h i o a

data NetworkStat = NetworkStat {parameters :: Sum Int, layers :: DL.DList LayerInfo}
  deriving (Show, Eq, Ord, Generic)
  deriving (Semigroup, Monoid) via Generically NetworkStat

data LayerInfo = AffL !Int | LinL !Int | ActL Activation | BatchL !Int | LayerNormL !Int
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
    (LayerNormRP {} :: RecParams _ i o a) ->
      NetworkStat
        { parameters = Sum $ dimVal @i * dimVal @o
        , layers = DL.singleton $ LayerNormL $ dimVal @o
        }

instance
  ( NFData a
  , U.Unbox a
  , forall l x y. NFData (h l x y a)
  ) =>
  NFData (Network h i ls o a)
  where
  rnf Output = ()
  rnf (h :- hs) = rnf h `seq` rnf hs
  rnf (h ::- hs) = rnf h `seq` rnf hs
  {-# INLINE rnf #-}

instance NFData (Weights l i o a) where
  rnf (AffW mat vec) = rnf mat `seq` rnf vec
  rnf (LinW mat) = rnf mat
  rnf ActW = ()
  rnf (BatW vec vec') = rnf vec `seq` rnf vec'
  rnf (LayerNormW vec vec') = rnf vec `seq` rnf vec'
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
  rnf LayerNormRP = ()
  {-# INLINE rnf #-}

getNetworkWith ::
  ( Persist a
  , U.Unbox a
  , KnownNat i
  , KnownNat o
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  ) =>
  NetworkShape i xs o ->
  Get (Network h i xs o a)
getNetworkWith shape = do
  validateNetworkHeader shape
  getNetworkBodyWith shape

validateNetworkHeader :: (KnownNat i, KnownNat o) => NetworkShape i xs o -> Get ()
validateNetworkHeader shape = do
  let expected = toNetworkHeader shape
  hdr <- get
  when (hdr /= expected) $
    fail $
      "validateNetworkHeader: network header mismatch (expected, got) = "
        <> show (expected, hdr)

getNetworkBodyWith ::
  ( Persist a
  , U.Unbox a
  , KnownNat o
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  ) =>
  NetworkShape i xs o ->
  Get (Network h i xs o a)
getNetworkBodyWith IsOutput = pure Output
getNetworkBodyWith (IsCons _ ns') = (:-) <$> get <*> getNetworkBodyWith ns'
getNetworkBodyWith (IsSkip sh ns') =
  give sh $
    (::-) <$> getNetworkBodyWith sh <*> getNetworkBodyWith ns'

putNetworkWithHeader ::
  forall h i xs o a.
  ( KnownNat i
  , forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  ) =>
  Network h i xs o a ->
  Put ()
putNetworkWithHeader net = withKnownNetwork net $ do
  put $ toNetworkHeader $ networkShape @i @xs @o
  putNetworkBody net

putNetworkBody ::
  (forall l x y. KnownLayerKind l x y => Persist (h l x y a)) =>
  Network h i ls o a ->
  Put ()
putNetworkBody net =
  getAp $ foldMapNetwork (Ap . put) net

instance
  ( forall l x y. KnownLayerKind l x y => Persist (h l x y a)
  , KnownNat i
  , KnownNat o
  , Persist a
  , U.Unbox a
  ) =>
  Persist (SomeNetwork h i o a)
  where
  put (MkSomeNetwork net) = withKnownNetwork net $ put net
  {-# INLINE put #-}
  get = do
    MkSomeNetworkShape (shape :: NetworkShape i' xs o') <- get
    Refl <-
      maybe
        ( fail $
            "get/SomeNeuralNetwork: input mismatched (expected, got) = "
              <> show (dimVal @i, dimVal @i')
        )
        pure
        $ testEquality (typeRep @i) (typeRep @i')
    Refl <-
      maybe
        ( fail $
            "get/SomeNeuralNetwork: output mismatched (expected, got) = "
              <> show (dimVal @o, dimVal @o')
        )
        pure
        $ testEquality (typeRep @o) (typeRep @o')
    give shape $ MkSomeNetwork <$> getNetworkWith shape
  {-# INLINE get #-}

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
    SAff -> AffW <$> get <*> get
    SLin -> LinW <$> get
    (SAct _) -> pure ActW
    SBN -> BatW <$> get <*> get
    SLN -> LayerNormW <$> get <*> get
  put (AffW mat vec) = put mat *> put vec
  put (LinW mat) = put mat
  put ActW = pure ()
  put (BatW vec vec') = put vec *> put vec'
  put (LayerNormW vec vec') = put vec *> put vec'
  {-# INLINE put #-}

instance
  (KnownLayerKind l i o, Persist a, U.Unbox a) =>
  Persist (RecParams l i o a)
  where
  get = case sLayerKind @l @i @o of
    SAff -> pure AffRP
    SLin -> pure LinRP
    SAct sact -> pure $ ActRP sact
    SBN -> BatRP <$> get <*> get
    SLN -> pure LayerNormRP
  put AffRP = pure ()
  put LinRP = pure ()
  put (ActRP _) = pure ()
  put (BatRP vec vec') = put vec *> put vec'
  put LayerNormRP = pure ()
  {-# INLINE put #-}
