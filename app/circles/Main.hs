{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module Main (main) where

import Control.Applicative (optional, (<**>))
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Lens hiding (Snoc)
import Control.Monad ((<=<))
import Data.DList (DList)
import qualified Data.DList as DL
import Data.Foldable (forM_)
import qualified Data.List as List
import Data.List.NonEmpty (NonEmpty (..))
import qualified Data.List.NonEmpty as NE
import Data.List.Split (splitOn)
import Data.Maybe (fromMaybe)
import Data.Monoid
import Data.Proxy (Proxy (..))
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.HigherKinded
import Diagrams.Backend.Rasterific
import Diagrams.Prelude (Diagram, alignB, alignT, bg, black, blend, centerXY, fc, green, lc, mkHeight, orange, p2, pad, scale, strokeOpacity, white, (===), (|||))
import qualified Diagrams.Prelude as Dia
import GHC.TypeNats
import Generic.Data
import Linear
import Linear.Affine
import Linear.V (V)
import Numeric.Natural (Natural)
import qualified Options.Applicative as Opts
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random.Stateful (globalStdGen)
import Text.Printf (printf)
import Text.Read (readMaybe)

main :: IO ()
main = do
  cmd <- Opts.execParser cmdP
  case cmd of
    Circles opts -> dualCircleTest opts
    Spirals opts -> dualSpiralTest opts

data Cmd = Circles Opts | Spirals Opts
  deriving (Show, Eq, Ord, Generic)

cmdP :: Opts.ParserInfo Cmd
cmdP =
  Opts.info (p <**> Opts.helper) $
    mconcat
      [ Opts.header "circles - hidden layer demo (Day 2)"
      , Opts.progDesc "Binary point classification with hidden layers"
      ]
  where
    p =
      Opts.subparser $
        mconcat
          [ Opts.command "circles" $
              Opts.info (Circles <$> optsP) (Opts.progDesc "Classifies dual circle datasets")
          , Opts.command "spirals" $
              Opts.info (Spirals <$> optsP) (Opts.progDesc "Classifies dual spiral datasets")
          ]

data Opts = Opts
  { epochs :: !Int
  , gamma :: !Double
  , layers :: NonEmpty (NonEmpty Natural)
  }
  deriving (Show, Eq, Ord)

optsP :: Opts.Parser Opts
optsP = do
  epochs <-
    Opts.option Opts.auto $
      Opts.short 'n'
        <> Opts.value 500
        <> Opts.showDefault
        <> Opts.metavar "N"
        <> Opts.help "# of epochs"
  gamma <-
    Opts.option Opts.auto $
      Opts.long "gamma"
        <> Opts.short 'g'
        <> Opts.value 0.001
        <> Opts.metavar "GAMMA"
        <> Opts.showDefault
        <> Opts.help "Learning rate"
  layers <-
    fmap (fromMaybe $ (128 :| []) :| []) $
      optional $
        NE.some1 $
          Opts.option (Opts.maybeReader intsP) $
            Opts.long "layers"
              <> Opts.short 'L'
  pure Opts {..}

intsP :: String -> Maybe (NonEmpty Natural)
intsP = NE.nonEmpty <=< mapM readMaybe . splitOn ","

workDir, circleWorkDir, spiralWorkDir :: FilePath
workDir = "workspace"
circleWorkDir = workDir </> "circles"
spiralWorkDir = workDir </> "spirals"

savePointImage :: FilePath -> U.Vector ClusteredPoint -> IO ()
savePointImage fp pts =
  renderRasterific fp (mkHeight 256) $
    drawClusteredPoints pts & bg white

mkPredictionImage ::
  (Dia.N b ~ Double, Dia.V b ~ V2, Dia.Renderable (Dia.Path V2 Double) b) =>
  NeuralNetwork V2 ls V1 Double ->
  Vector ClusteredPoint ->
  Diagram b
mkPredictionImage nn pts =
  mconcat
    [ drawClusteredPoints pts & lc black & strokeOpacity 1.0
    , pixelateScalarField
        64
        (view _x . evalNN nn . view _Point)
        (\α -> blend (min 1.0 $ max 0.0 α) green orange)
        (p2 (-1.25, -1.25))
        (p2 (1.25, 1.25))
    ]

savePredictionComparisonImage ::
  FilePath ->
  NeuralNetwork V2 ls V1 Double ->
  (String, Vector ClusteredPoint) ->
  (String, Vector ClusteredPoint) ->
  IO ()
savePredictionComparisonImage fp nn (lab0, pts0) (lab1, pts1) =
  renderRasterific fp (mkHeight 256) $
    ( ( (mkPredictionImage nn pts0 & centerXY & alignB)
          === (texterific lab0 & scale 0.2 & fc white & centerXY & pad 1.5 & alignT)
      )
        ||| ( (mkPredictionImage nn pts1 & centerXY & alignB)
                === (texterific lab1 & scale 0.2 & fc white & centerXY & pad 1.1 & alignT)
            )
    )
      & centerXY
      & pad 1.1
      & bg green

data SomeNetSeedAux i where
  BaseNet :: Network ActivatorProxy i '[V1] V1 Double -> SomeNetSeedAux i
  MkSomeNet ::
    ( Describable Layer (l ': ls)
    , Applicative (WeightStack i (l ': ls) V1)
    ) =>
    Proxy l ->
    Proxy ls ->
    Network ActivatorProxy i (l ': ls) V1 Double ->
    SomeNetSeedAux i

data SomeNetSeed i where
  MkSomeNetSeed ::
    (Describable Layer ls, Applicative (WeightStack i ls V1)) =>
    Network ActivatorProxy i ls V1 Double ->
    SomeNetSeed i

withSomeNetwork ::
  NonEmpty Natural ->
  ( forall ls.
    (Describable Layer ls, Applicative (WeightStack V2 ls V1)) =>
    NeuralNetwork V2 ls V1 Double ->
    IO a
  ) ->
  IO a
withSomeNetwork layers f = case toSomeNetworkSeed layers of
  MkSomeNetSeed net ->
    f =<< randomNetwork globalStdGen net

toSomeNetworkSeed :: NonEmpty Natural -> SomeNetSeed V2
toSomeNetworkSeed = fromAux . go . NE.toList
  where
    fromAux (BaseNet net) = MkSomeNetSeed net
    fromAux (MkSomeNet _ _ net) = MkSomeNetSeed net
    go :: Applicative v => [Natural] -> SomeNetSeedAux v
    go [] = BaseNet $ sigmoidP @V1 :- Output
    go (n : ns) = case someNatVal n of
      SomeNat (_ :: Proxy n) ->
        case go @(V n) ns of
          BaseNet net ->
            MkSomeNet Proxy Proxy $ reLUP :- net
          MkSomeNet (_ :: Proxy l) (_ :: Proxy ls) net ->
            MkSomeNet (Proxy @(V n)) (Proxy @(l ': ls)) $
              reLUP @(V n) :- net

data NetworkDescr = NetworkDescr {hiddenLayers :: !(Sum Int, DList Int), neurons :: !(Sum Int), parameters :: !(Sum Int)}
  deriving (Show, Eq, Ord, Generic)
  deriving (Semigroup, Monoid) via Generically NetworkDescr

class Describable h ls where
  describe ::
    (Applicative i, Foldable i) =>
    Network h i ls o a ->
    NetworkDescr

instance Describable h '[] where
  describe (Output :: Network h i '[] o a) =
    mempty {neurons = Sum $ length $ pure @i ()}

instance
  (forall x y. (Foldable x, Foldable y) => Foldable (h x y)) =>
  Describable h '[o]
  where
  describe (h :- Output) =
    mempty {parameters = Sum $ length h}

instance
  ( forall x y. (Foldable x, Foldable y) => Foldable (h x y)
  , Describable h (l' ': ls)
  ) =>
  Describable h (l ': l' ': ls)
  where
  describe (h :- rest) =
    let numNeuron = length $ pure @l ()
     in mempty
          { parameters = Sum $ length h
          , neurons = Sum numNeuron
          , hiddenLayers = (1, DL.singleton numNeuron)
          }
          <> describe rest

putNetworkInfo ::
  ( Applicative i
  , Foldable i
  , Describable h ls
  ) =>
  Network h i ls o a ->
  IO ()
putNetworkInfo net = do
  let NetworkDescr {..} = describe net
  putStrLn $
    printf
      "** %d hidden layer(s) (%s), %d neurons (%d parameters)"
      (getSum $ fst hiddenLayers)
      (showDim $ DL.toList $ snd hiddenLayers)
      (getSum neurons)
      (getSum parameters)

showDim :: Show a => [a] -> String
showDim = List.intercalate "x" . map show

type Snoc :: [k] -> k -> [k]
type family Snoc xs x where
  Snoc '[] x = '[x]
  Snoc (x ': xs) y = x ': Snoc xs y

dualCircleTest :: Opts -> IO ()
dualCircleTest Opts {..} = do
  createDirectoryIfMissing True circleWorkDir
  trainSet <- evaluate . force =<< dualCircles globalStdGen 200 0.6 0.1
  testSet <- evaluate . force =<< dualCircles globalStdGen 100 0.6 0.1

  savePointImage (circleWorkDir </> "train.png") trainSet
  savePointImage (circleWorkDir </> "test.png") testSet

  putStrLn $ replicate 20 '-'
  putStrLn $
    printf "* Circle isolation, Circle isolation, %d epochs, learning rate = %f" epochs gamma

  forM_ layers $ \lay -> withSomeNetwork lay $ \net0 -> do
    putNetworkInfo net0

    putStrLn $ printf "Initial training accuracy (GD): %f" $ predictionAccuracy net0 trainSet * 100
    putStrLn $ printf "Initial validation accuracy (GD): %f" $ predictionAccuracy net0 testSet * 100
    let net' = trainByGradientDescent gamma epochs trainSet net0

    savePredictionComparisonImage
      ( circleWorkDir
          </> printf "predict-gd-%s.png" (showDim $ NE.toList lay)
      )
      net'
      ("Train", trainSet)
      ("Test", testSet)

    putStrLn $ printf "Training accuracy (GD): %f" $ predictionAccuracy net' trainSet * 100
    putStrLn $ printf "Validation accuracy (GD): %f" $ predictionAccuracy net' testSet * 100

dualSpiralTest :: Opts -> IO ()
dualSpiralTest Opts {..} = do
  createDirectoryIfMissing True spiralWorkDir
  trainSet <- evaluate . force =<< dualSpirals globalStdGen 400 0.05
  testSet <- evaluate . force =<< dualSpirals globalStdGen 200 0.05
  putStrLn ""
  putStrLn $ replicate 20 '-'
  putStrLn $
    printf
      "* Dual spiral classification, %d epochs, learn rate = %f"
      epochs
      gamma
  savePointImage (spiralWorkDir </> "train.png") trainSet
  savePointImage (spiralWorkDir </> "test.png") testSet

  forM_ layers $ \lay -> withSomeNetwork lay $ \net0 -> do
    putNetworkInfo net0
    netGD <- evaluate $ trainByGradientDescent gamma epochs trainSet net0
    putStrLn $ printf "Training accuracy (GD): %f" $ predictionAccuracy netGD trainSet * 100
    putStrLn $ printf "Validation accuracy (GD): %f" $ predictionAccuracy netGD testSet * 100

    savePredictionComparisonImage
      ( spiralWorkDir
          </> printf "predict-gd-%s.png" (showDim $ NE.toList lay)
      )
      netGD
      ("Train", trainSet)
      ("Test", testSet)

    netAdam <- evaluate $ trainByAdam gamma AdamParams {beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8} epochs trainSet net0

    putStrLn $ printf "Training accuracy (Adam): %f" $ predictionAccuracy netAdam trainSet * 100
    putStrLn $ printf "Validation accuracy (Adam): %f" $ predictionAccuracy netAdam testSet * 100

    savePredictionComparisonImage
      ( spiralWorkDir
          </> printf "predict-adam-%s.png" (showDim $ NE.toList lay)
      )
      netAdam
      ("Train", trainSet)
      ("Test", testSet)
