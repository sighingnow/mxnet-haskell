-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Test suite for mxnet package.
--
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}

import qualified Data.Vector.Storable as V

import           Test.Tasty
import           Test.Tasty.HUnit

import           MXNet.Core.Base

main :: IO ()
main = mxListAllOpNames >> defaultMain mxnetTest

mxnetTest :: TestTree
mxnetTest = testGroup "MXNet Test Suite"
    [ hmapTest
    , ndarrayTest
    , symbolTest
    ]

hmapTest :: TestTree
hmapTest = testGroup "HMap"
    [ testCase "Get after add" $ do
        let expected = 1
            got = get @"a" (add @"a" (1 :: Int) nil)
        assertEqual "get after add" expected got
    ]

ndarrayTest :: TestTree
ndarrayTest = testGroup "NDArray"
    [ testCaseSteps "NDArray basic operation" $ \step -> do
        step "preparing ndarray"
        let sh = [2, 3, 4, 5]
            p = product sh
        arr <- array sh [1 .. fromIntegral $ product sh] :: IO (NDArray Float)

        step "shape and size"
        (d, sh') <- ndshape arr
        assertEqual "dimension should coincide" 4 d
        assertEqual "shape should coincide" sh sh'
        s <- ndsize arr
        assertEqual "size should coincide" p s

        step "arithmetic operators"
        let b = (arr + arr) .* 3 ./ 2
        r <- V.sum <$> items b
        let expected = 3 * (p * (p+1) `div` 2)
        assertEqual "sum of elements should be as expected" expected (round r)
        
        step "comparison with scalar"
        let b = (_Maximum' arr 1000)
        r <- V.sum <$> items b
        let expected = 1000 * p
        assertEqual "_Maximum' should set the whole ndarray" expected (round r)

    , testCaseSteps "NDArray linear algebra" $ \step -> do
        step "preparing ndarray"
        a <- array [2, 3] [1 .. 6]

        step "ndarray dot product"
        let b = a `dot` transpose a
        expected1 <- array [2, 2] [14, 32, 32, 77] :: IO (NDArray Float)
        assertEqual "a `dot` transpose a" expected1 b
        let c = transpose a `dot` a
        expected2 <- array [3, 3] [17, 22, 27, 22, 29, 36, 27, 36, 45] :: IO (NDArray Float)
        assertEqual "transpose a `dot` a" expected2 c

    , testCaseSteps "NDArray activation" $ \step -> do
        step "preparing ndarray"
        a <- array [4] [-0.5, -0.1, 0.1, 0.5]

        step "relu activation"
        let r = activation a "relu"
        expected <- array [4] [0.0, 0.0, 0.1, 0.5] :: IO (NDArray Float)
        assertEqual "relu activation" expected r

        step "sigmoid activation"
        let r = activation a "sigmoid"
        expected <- array [4] [0.37754068, 0.4750208, 0.52497917, 0.62245935] :: IO (NDArray Float)
        assertEqual "sigmoid activation" expected r

        step "softrelu activation"
        let r = activation a "softrelu"
        expected <- array [4] [0.47407699, 0.64439672, 0.74439669, 0.97407699] :: IO (NDArray Float)
        assertEqual "softrelu activation" expected r

        step "tanh activation"
        let r = activation a "tanh"
        expected <- array [4] [-0.46211717, -0.099667996, 0.099667996, 0.46211717] :: IO (NDArray Float)
        assertEqual "tanh activation" expected r

        step "softmax activation"
        let r = softmaxActivation a
        expected <- array [4] [0.1422025, 0.21214119, 0.25910985, 0.38654646] :: IO (NDArray Float)
        assertEqual "softmax activation" expected r

        step "leakyReLU activation"  -- for leakyReLU, input must be a multiple dimensions ndarray.
        a' <- array [1, 4] [-0.5, -0.1, 0.1, 0.5]
        let r = leakyReLU a' "leaky" 
        expected <- array [1, 4] [-0.125, -0.025, 0.1, 0.5] :: IO (NDArray Float)
        assertEqual "leakyReLU activation" expected r
    ]

symbolTest :: TestTree
symbolTest = testGroup "Symbol"
    [ testCaseSteps "Symbol basic operation" $ \step -> do
        step "preparing symbol"
        a <- variable "a" :: IO (Symbol Float)

        step "get name"
        a' <- getName a
        assertEqual "get symbol name" "a" a'

    , testCaseSteps "Symbol bind ndarray data" $ \step -> do
        step "preparing symbol and ndarray"
        a <- variable "a" :: IO (Symbol Float)
        b <- variable "b" :: IO (Symbol Float)
        arr1 <- array [2, 3] [1 .. 6]
        arr2 <- array [2, 3] [5 .. 10]

        step "bind and arithmetic operators"
        let c = (a + b .+ 2) ./ 2
        expected <- array [2, 3] [4 .. 9]
        exec <- bind c contextCPU [ ("a", arr1)
                                  , ("b", arr2) ]
        forward exec False
        [r] <- getOutputs exec
        assertEqual "bind and arithmetics with scalar" expected r

        step "bind and linear algebra"
        let c = a `dot` transpose b
        expected <- array [2, 2] [38, 56, 92, 137]
        exec <- bind c contextCPU [ ("a", arr1)
                                  , ("b", arr2) ]
        forward exec False
        [r] <- getOutputs exec
        assertEqual "bind and a `dot` transpose b" expected r
        let c = transpose a `dot` b
        expected <- array [3, 3] [ 37, 42, 47
                                 , 50, 57, 64
                                 , 63, 72, 81 ]
        exec <- bind c contextCPU [ ("a", arr1)
                                  , ("b", arr2) ]
        forward exec False
        [r] <- getOutputs exec
        assertEqual "bind and transpose a `dot` b" expected r

        step "bind and activation"
        let c = softmaxActivation a
        expected <- array [2, 3] [ 0.09003057, 0.24472848, 0.66524094
                                 , 0.09003057, 0.24472848, 0.66524094 ]
        exec <- bind c contextCPU [ ("a", arr1) ]
        forward exec False
        [r] <- getOutputs exec
        assertEqual "bind and softmax activation" expected r
        let c = leakyReLU a "leaky"
        expected <- array [2, 3] [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ]
        exec <- bind c contextCPU [ ("a", arr1) ]
        forward exec False
        [r] <- getOutputs exec
        assertEqual "bind and leakyReLU activation" expected r
    ]

