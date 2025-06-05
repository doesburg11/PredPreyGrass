using System;

namespace UnityEngine.U2D.Animation
{
    internal static class SpriteLibraryUtility
    {
        // Allow delegate override for test
        internal static Func<string, int> GetStringHash = Bit30Hash_GetStringHash;

        /// <summary>
        /// Used to convert Sprite Key to the new Sprite Hash.
        /// </summary>
        /// <param name="input">Sprite Key to convert</param>
        /// <returns>A 30-bit long hash.</returns>
        internal static int Convert32BitTo30BitHash(int input)
        {
            var output = PreserveFirst30Bits(input);
            return output;
        }

        static int Bit30Hash_GetStringHash(string value)
        {
#if DEBUG_GETSTRINGHASH_CLASH
            if (value == "abc" || value == "123")
                value = "abc";
#endif
            var hash = Animator.StringToHash(value);
            hash = PreserveFirst30Bits(hash);
            return hash;
        }

        static int PreserveFirst30Bits(int input)
        {
            const int mask = 0x3FFFFFFF;
            return input & mask;
        }

        internal static long GenerateHash()
        {
            var hash = DateTime.Now.Ticks;
            return hash;
        }
    }
}
