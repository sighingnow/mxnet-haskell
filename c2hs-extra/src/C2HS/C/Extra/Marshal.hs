-----------------------------------------------------------
-- |
-- module:                      C2HS.C.Extra.Marshal
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Convenient marshallers for complicate C types.
--
module C2HS.C.Extra.Marshal
    ( peekIntegral
    , peekString
    , peekStringArray
    , withStringArray
    ) where

import Foreign.C.String ( peekCString, withCString )
import Foreign.C.Types ( CChar, CInt, CUInt )
import Foreign.Marshal.Array ( peekArray, withArray )
import Foreign.Ptr ( Ptr )
import Foreign.Storable ( Storable, peek )

-- | Peek from pointer then cast to another integral type.
peekIntegral :: (Integral a, Storable a, Integral b) => Ptr a -> IO b
peekIntegral = (fromIntegral <$>) . peek

{-# SPECIALIZE peekIntegral :: Ptr CInt -> IO Int #-}

-- | Peek string from a two-dimension pointer of CChar.
peekString :: Ptr (Ptr CChar) -> IO String
peekString p = peek p >>= peekCString

-- | Peek an array of string and the result's length is given.
peekStringArray :: Integral n => n -> Ptr (Ptr CChar) -> IO [String]
peekStringArray n p = peekArray (fromIntegral n) p >>= mapM peekCString

{-# SPECIALIZE peekStringArray :: Int -> Ptr (Ptr CChar) -> IO [String] #-}
{-# SPECIALIZE peekStringArray :: CUInt -> Ptr (Ptr CChar) -> IO [String] #-}

-- | Use an array of string as argument, usually used to pass multiple names to C
-- functions.
withStringArray :: [String] -> (Ptr (Ptr CChar) -> IO a) -> IO a
withStringArray ss f = do
    ps <- mapM (\s -> withCString s return) ss
    withArray ps f
