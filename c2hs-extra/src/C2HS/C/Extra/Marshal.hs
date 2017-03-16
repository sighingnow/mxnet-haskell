-----------------------------------------------------------
-- |
-- module:                      C2HS.C.Extra.Marshal
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Convenient marshallers for complicate C types.
--
{-# LANGUAGE CPP #-}
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif

module C2HS.C.Extra.Marshal
    ( peekIntegral
    , peekString
    , peekStringArray
    , withStringArray
    , peekIntegralArray
    , withIntegralArray
    ) where

import Foreign.C.String ( peekCString, withCString )
import Foreign.C.Types ( CChar, CInt, CUInt )
import Foreign.Marshal.Array ( peekArray, withArray )
import Foreign.Ptr ( Ptr, nullPtr )
import Foreign.Storable ( Storable, peek )

-- | Peek from pointer then cast to another integral type.
peekIntegral :: (Integral a, Storable a, Integral b) => Ptr a -> IO b
peekIntegral p = if p == nullPtr
                    then return 0
                    else fromIntegral <$> peek p

{-# SPECIALIZE peekIntegral :: Ptr CInt -> IO Int #-}

-- | Peek string from a two-dimension pointer of CChar.
peekString :: Ptr (Ptr CChar) -> IO String
peekString p = if p == nullPtr
                  then return ""
                  else peek p >>= \p' ->
                      if p' == nullPtr
                         then return ""
                         else peekCString p'

-- | Peek an array of String and the result's length is given.
peekStringArray :: Integral n => n -> Ptr (Ptr CChar) -> IO [String]
peekStringArray 0 _ = return []
peekStringArray n p = if p == nullPtr
                         then return []
                         else peekArray (fromIntegral n) p >>=
                             mapM (\p' -> if p' == nullPtr
                                             then return ""
                                             else peekCString p')

{-# SPECIALIZE peekStringArray :: Int -> Ptr (Ptr CChar) -> IO [String] #-}
{-# SPECIALIZE peekStringArray :: CUInt -> Ptr (Ptr CChar) -> IO [String] #-}

-- | Use an array of String as argument, usually used to pass multiple names to C
-- functions.
withStringArray :: [String] -> (Ptr (Ptr CChar) -> IO a) -> IO a
withStringArray [] f = f nullPtr
withStringArray ss f = do
    ps <- mapM (\s -> withCString s return) ss
    withArray ps f

-- | Peek an array of integral values and the result's length is given.
peekIntegralArray :: (Integral n, Integral m, Storable m) => Int -> Ptr m -> IO [n]
peekIntegralArray n p = (map fromIntegral) <$> peekArray n p

{-# SPECIALIZE peekIntegralArray :: Int -> Ptr CInt -> IO [Int] #-}
{-# SPECIALIZE peekIntegralArray :: Int -> Ptr CUInt -> IO [Int] #-}

-- | Use an array of Integral as argument.
withIntegralArray :: (Integral a, Integral b, Storable b) => [a] -> (Ptr b -> IO c) -> IO c
withIntegralArray ns f = do
    let ns' = fmap fromIntegral ns
    withArray ns' f

{-# SPECIALIZE withIntegralArray :: [Int] -> (Ptr CInt -> IO c) -> IO c #-}
{-# SPECIALIZE withIntegralArray :: [CInt] -> (Ptr CInt -> IO c) -> IO c #-}
{-# SPECIALIZE withIntegralArray :: [CUInt] -> (Ptr CUInt -> IO c) -> IO c #-}
