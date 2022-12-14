{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module Data.Format.MNIST (
  -- * Types and constructors
  Digit (D0, D1, D2, D3, D4, D5, D6, D7, D8, D9),
  digit,
  getDigit,
  LabelFileHeader (..),
  ImageFileHeader (..),
  MNISTException (..),
  Pixel,
  Image,
  fromGrayscaleImage,

  -- * Parsers and Formatters

  -- ** Label File

  -- *** Parsers
  labelFileMagicNumber,
  labelFileHeaderP,
  digitP,
  labelFileP,
  parseLabelFileS,

  -- *** Formatters
  formatLabelFileHeader,
  formatDigit,
  streamLabelFile,

  -- ** Image File

  -- *** Parsers
  imageFileMagicNumber,
  imageFileHeaderP,
  pixelP,
  imageP,
  imageFileP,
  parseImageFileS,

  -- *** Formatters
  formatImageFileHeader,
  formatImage,
  streamImageFile,
) where

import Control.DeepSeq (NFData)
import Control.Exception (Exception, throwIO)
import Control.Monad (guard, void, (>=>))
import Control.Monad.IO.Class (MonadIO (..))
import Control.Monad.Trans.Class (lift)
import qualified Data.Attoparsec.Binary as BA
import Data.Attoparsec.ByteString (Parser, (<?>))
import qualified Data.Attoparsec.ByteString as A
import qualified Data.Attoparsec.ByteString.Streaming as AQ
import qualified Data.ByteString.Builder as B
import Data.Coerce (coerce)
import Data.Function ((&))
import Data.Functor.Of (Of (..))
import Data.Massiv.Array (Sz (..))
import qualified Data.Massiv.Array as M
import qualified Data.Massiv.Array as MS
import qualified Data.Massiv.Array.IO as MIO
import Data.Maybe (fromMaybe)
import qualified Data.Vector.Unboxed as U
import Data.Vector.Unboxed.Deriving (derivingUnbox)
import Data.Word (Word32, Word8)
import GHC.Generics (Generic)
import qualified Streaming.ByteString as Q
import qualified Streaming.Prelude as S

newtype Digit = Digit Word8
  deriving (Show, Eq, Ord)

derivingUnbox
  "Digit"
  [t|Digit -> Word8|]
  [|coerce|]
  [|coerce|]

getDigit :: Digit -> Word8
getDigit = coerce

digit :: Word8 -> Maybe Digit
digit n = Digit n <$ guard (n <= 9)

pattern D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 :: Digit
pattern D0 = Digit 0
pattern D1 = Digit 1
pattern D2 = Digit 2
pattern D3 = Digit 3
pattern D4 = Digit 4
pattern D5 = Digit 5
pattern D6 = Digit 6
pattern D7 = Digit 7
pattern D8 = Digit 8
pattern D9 = Digit 9

{-# COMPLETE D0, D1, D2, D3, D4, D5, D6, D7, D8, D9 :: Digit #-}

instance Enum Digit where
  toEnum = fromMaybe (error "Digit out of bound") . digit . toEnum
  {-# INLINE toEnum #-}
  fromEnum = fromEnum . getDigit
  {-# INLINE fromEnum #-}
  succ (Digit n)
    | n <= 8 = Digit $ succ n
    | otherwise = error "succ: out of bound"
  {-# INLINE succ #-}
  pred (Digit n)
    | 1 <= n && n <= 9 = Digit $ pred n
    | otherwise = error "pred: out of bounds"
  {-# INLINE pred #-}
  enumFrom (Digit n) = coerce [n .. 9]
  {-# INLINE enumFrom #-}
  enumFromTo (Digit n) (Digit m) = coerce [n .. min m 9]
  {-# INLINE enumFromTo #-}
  enumFromThen (Digit n) (Digit m) =
    coerce $ enumFromThenTo n m 9
  {-# INLINE enumFromThen #-}
  enumFromThenTo (Digit n) (Digit m) (Digit k) =
    coerce $ enumFromThenTo n m (min k 9)

instance Bounded Digit where
  minBound = D0
  {-# INLINE minBound #-}
  maxBound = D9
  {-# INLINE maxBound #-}

labelFileMagicNumber :: Word32
labelFileMagicNumber = 2049

newtype LabelFileHeader = LabelFileHeader {labelCount :: Word32}
  deriving (Show, Eq, Ord, Generic)
  deriving newtype (NFData)

labelFileHeaderP :: Parser LabelFileHeader
labelFileHeaderP =
  (BA.word32be labelFileMagicNumber <?> "label magic number")
    *> (LabelFileHeader <$> BA.anyWord32be <?> "label count")
    <?> "label file header"

formatLabelFileHeader :: LabelFileHeader -> B.Builder
{-# INLINE formatLabelFileHeader #-}
formatLabelFileHeader hdr =
  B.word32BE labelFileMagicNumber <> B.word32BE (labelCount hdr)

digitP :: Parser Digit
digitP = do
  w <- A.anyWord8 <?> "digit"
  guard $ w <= 9
  pure $ Digit w

formatDigit :: Digit -> B.Builder
{-# INLINE formatDigit #-}
formatDigit = coerce B.word8

labelFileP :: Parser (U.Vector Digit)
labelFileP = do
  lh <- labelFileHeaderP
  U.replicateM (fromIntegral $ labelCount lh) digitP

newtype MNISTException = ParseError ([String], String)
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (Exception)

parseE :: MonadIO m => Parser a -> Q.ByteStream m x -> m (a, Q.ByteStream m x)
parseE p =
  AQ.parse p >=> \case
    (Left err, _) -> liftIO $ throwIO $ ParseError err
    (Right hdr, rest) -> pure (hdr, rest)

parseLabelFileS ::
  MonadIO m =>
  Q.ByteStream m r ->
  m (LabelFileHeader, S.Stream (Of Digit) m ())
parseLabelFileS =
  parseE labelFileHeaderP
    >=> \(hdr@LabelFileHeader {..}, rest) ->
      pure
        (hdr, void $ parseExactN (fromIntegral labelCount) digitP rest)

streamLabelFile :: forall m r. (MonadIO m) => LabelFileHeader -> S.Stream (Of Digit) m r -> Q.ByteStream m r
streamLabelFile hdr s = do
  Q.toStreamingByteString $ formatLabelFileHeader hdr
  s
    & S.splitAt (fromIntegral $ labelCount hdr)
    & S.subst (Q.toStreamingByteString @m . formatDigit)
    & Q.concat
    & S.drained

imageFileMagicNumber :: Word32
imageFileMagicNumber = 2051

data ImageFileHeader = ImageFileHeader {imageCount, rowCount, columnCount :: !Word32}
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (NFData)

imageFileHeaderP :: Parser ImageFileHeader
imageFileHeaderP =
  ImageFileHeader
    <$ (BA.word32be imageFileMagicNumber <?> "image file magic number")
    <*> (BA.anyWord32be <?> "number of images")
    <*> (BA.anyWord32be <?> "number of rows")
    <*> (BA.anyWord32be <?> "number of columns")
    <?> "image file header"

formatImageFileHeader :: ImageFileHeader -> B.Builder
formatImageFileHeader hdr =
  B.word32BE imageFileMagicNumber
    <> B.word32BE (imageCount hdr)
    <> B.word32BE (rowCount hdr)
    <> B.word32BE (columnCount hdr)

type Pixel = Word8

pixelP :: Parser Pixel
pixelP = A.anyWord8 <?> "pixel"

formatPixel :: Pixel -> B.Builder
formatPixel = B.word8

type Image = M.Array M.U M.Ix2 Pixel

{- |
Decodes hand-written grayscale image, interpreting
black as 255, white as 0.
-}
fromGrayscaleImage :: MIO.Image M.S (MIO.Y' MIO.SRGB) Word8 -> Image
{-# INLINE fromGrayscaleImage #-}
fromGrayscaleImage = M.computeP . M.map (\(MIO.PixelY' c) -> 255 - c)

imageP :: ImageFileHeader -> Parser Image
imageP ImageFileHeader {..} =
  M.makeArrayA
    (Sz2 (fromIntegral rowCount) (fromIntegral columnCount))
    (const pixelP)
    <?> "image"

formatImage :: Image -> B.Builder
{-# INLINE formatImage #-}
formatImage = M.foldMono formatPixel

imageFileP :: Parser (MS.Vector MS.DS Image)
imageFileP = flip (<?>) "image file" $ do
  h@ImageFileHeader {..} <- imageFileHeaderP
  MS.sreplicateM (fromIntegral imageCount) $ imageP h

parseImageFileS ::
  MonadIO m =>
  Q.ByteStream m x ->
  m (ImageFileHeader, S.Stream (Of Image) m ())
{-# INLINE parseImageFileS #-}
parseImageFileS =
  parseE imageFileHeaderP >=> \(h@ImageFileHeader {..}, rest) ->
    pure (h, void $ parseExactN (fromIntegral imageCount) (imageP h) rest)

streamImageFile ::
  MonadIO m =>
  ImageFileHeader ->
  S.Stream (Of Image) m r ->
  Q.ByteStream m r
streamImageFile hdr@ImageFileHeader {..} s = do
  Q.toStreamingByteString (formatImageFileHeader hdr)
  s
    & S.splitAt (fromIntegral imageCount)
    & S.subst (Q.toStreamingByteString . formatImage)
    & S.drained
    & Q.concat

parseExactN :: MonadIO m => Int -> Parser a -> Q.ByteStream m x -> S.Stream (Of a) m (Q.ByteStream m x)
{-# INLINE parseExactN #-}
parseExactN = (. parseE) . unfoldrExactNS

unfoldrExactNS :: Monad m => Int -> (s -> m (a, s)) -> s -> S.Stream (Of a) m s
{-# INLINE unfoldrExactNS #-}
unfoldrExactNS n0 f = loop n0
  where
    loop 0 s = pure s
    loop !n s = do
      (a, s') <- lift $ f s
      S.yield a
      loop (n - 1) s'
