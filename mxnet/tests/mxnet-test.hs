-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Test suite for mxnet package.
--
import           Control.Monad
import           Test.Tasty
import           Test.Tasty.QuickCheck
import           Test.QuickCheck.Monadic

import           MXNet.Core.Base
import           MXNet.Core.NDArray

main :: IO ()
main = defaultMain mxnetTest >> void mxNotifyShutdown

mxnetTest :: TestTree
mxnetTest = testGroup "MXNet Test Suite"
    [ ndarrayTest
    ]

ndarrayTest :: TestTree
ndarrayTest = testGroup "NDArray"
    [ testProperty "NDArray shape should coincide" $ monadicIO $ do
        let shape = [2, 3, 4, 5]

        shape' <- run $ do
            array <- makeNDArray shape
            (_, shape') <- getNDArrayShape array
            return shape'

        stop $ shape === shape
    ]



