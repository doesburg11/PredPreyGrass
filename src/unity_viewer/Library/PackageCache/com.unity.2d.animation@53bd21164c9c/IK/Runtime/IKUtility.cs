using UnityEngine.Scripting.APIUpdating;

namespace UnityEngine.U2D.IK
{
    /// <summary>
    /// General utilities for 2D IK.
    /// </summary>
    [MovedFrom("UnityEngine.Experimental.U2D.IK")]
    public class IKUtility
    {
        /// <summary>
        /// Check if a transform is a descendent of another transform.
        /// </summary>
        /// <param name="transform">Transforms to check.</param>
        /// <param name="ancestor">Transform's ancestor.</param>
        /// <returns>Returns true if the transform is a descendent. False otherwise.</returns>
        public static bool IsDescendentOf(Transform transform, Transform ancestor)
        {
            Debug.Assert(transform != null, "Transform is null");

            var currentParent = transform.parent;

            while (currentParent)
            {
                if (currentParent == ancestor)
                    return true;

                currentParent = currentParent.parent;
            }

            return false;
        }

        /// <summary>
        /// Gets the depth of the transform's hierarchy.
        /// </summary>
        /// <param name="transform">Transform to check.</param>
        /// <returns>Integer value for hierarchy depth.</returns>
        public static int GetAncestorCount(Transform transform)
        {
            Debug.Assert(transform != null, "Transform is null");

            var ancestorCount = 0;

            while (transform.parent)
            {
                ++ancestorCount;

                transform = transform.parent;
            }

            return ancestorCount;
        }

        /// <summary>
        /// Gets the maximum chain count for a IKChain2D.
        /// </summary>
        /// <param name="chain">IKChain2D to query.</param>
        /// <returns>Integer value for the maximum chain count.</returns>
        public static int GetMaxChainCount(IKChain2D chain)
        {
            var maxChainCount = 0;

            if (chain.effector)
                maxChainCount = GetAncestorCount(chain.effector) + 1;

            return maxChainCount;
        }
    }
}