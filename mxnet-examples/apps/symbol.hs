-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Example of how to work with Symbol.
--
import           MXNet.Core.Base

main :: IO ()
main = do
    var <- variable "abde" :: IO (Symbol Float)
    getName var >>= putStrLn
