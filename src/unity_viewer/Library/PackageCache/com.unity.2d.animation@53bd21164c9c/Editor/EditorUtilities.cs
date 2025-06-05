using System;
using Unity.Mathematics;
using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal static class EditorUtilities
    {
        /// <summary>
        /// Checks if element exists in array independent of the order of X and Y.
        /// </summary>
        public static bool ContainsAny(this int2[] array, int2 element)
        {
            return Array.FindIndex(array, e =>
                (e.x == element.x && e.y == element.y) ||
                (e.y == element.x && e.x == element.y)) != -1;
        }

        public static int2[] ToInt2(Vector2Int[] source) => Array.ConvertAll(source, e => new int2(e.x, e.y));
        public static Vector2Int[] ToVector2Int(int2[] source) => Array.ConvertAll(source, e => new Vector2Int(e.x, e.y));
        public static float2[] ToFloat2(Vector2[] source) => Array.ConvertAll(source, e => (float2)e);
        public static Vector2[] ToVector2(float2[] source) => Array.ConvertAll(source, e => (Vector2)e);

        public static T[] CreateCopy<T>(T[] source) where T : struct
        {
            var copy = new T[source.Length];
            Array.Copy(source, copy, source.Length);
            return copy;
        }
    }
}