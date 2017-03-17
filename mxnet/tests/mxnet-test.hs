-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Test suite for mxnet package.
--
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}

import           Control.Monad
import           Test.Tasty
import           Test.Tasty.QuickCheck
import           Test.QuickCheck.Monadic

import           MXNet.Core.Base

main :: IO ()
main = mxListAllOpNames >> defaultMain mxnetTest

mxnetTest :: TestTree
mxnetTest = testGroup "MXNet Test Suite"
    [ hmapTest
    , ndarrayTest
    ]

hmapTest :: TestTree
hmapTest = testGroup "HMap"
    [ testProperty "Get after add" $
        get @"a" (add @"a" (1 :: Int) nil) === 1
    ]

ndarrayTest :: TestTree
ndarrayTest = testGroup "NDArray"
    [ testProperty "NDArray shape should coincide" $ monadicIO $ do
        let sh = [2, 3, 4, 5]
        sh' <- run $ do
            arr <- array sh [1..(2*3*4*5)] :: IO (NDArray Float)
            (_, sh'') <- ndshape arr
            return sh''
        stop $ sh === sh'
    , testProperty "NDArray reshape should keep size" $ monadicIO $ do
        let sh = [2, 3, 4, 5]
        s <- run $ do
            arr <- array sh [1..(2*3*4*5)] :: IO (NDArray Float)
            ndsize arr
        stop $ product sh === s
    ]


