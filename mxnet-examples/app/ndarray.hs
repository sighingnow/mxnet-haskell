-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Example of how to work with NDArray.
--

import           MXNet.Core.NDArray

main :: IO ()
main = makeNDArray [2,3,4] >>= getNDArrayShape >>= print
