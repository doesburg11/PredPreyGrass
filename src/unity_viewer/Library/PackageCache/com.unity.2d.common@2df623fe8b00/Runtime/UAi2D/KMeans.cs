using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace UnityEngine.U2D.Common.UAi
{

    // Calculate KMeans Clustering for any Dimensional Vector.
    // External Libaries utilizing this class are expected to wrap Burst
    internal static class kMeans
    {

        // GeneralizedMatrix data and centroid should be same dimension.
        private static float CalculateDistance(MatrixMxN<float> data, int dataIndex, MatrixMxN<float> centroid, int centroidIndex)
        {
            var sum = 0.0f;
            for (var i = 0; i < data.DimensionY; i++)
                sum += Mathf.Pow(centroid.Get(centroidIndex, i) - data.Get(dataIndex, i), 2);
            return Mathf.Sqrt(sum);
        }

        private unsafe static float CalculateClustering(MatrixMxN<float> data, NativeArray<int> clusters, ref MatrixMxN<float> means, ref NativeArray<int> centroids, int clusterCount, ref NativeArray<int> clusterItems)
        {

            UnsafeUtility.MemSet(NativeArrayUnsafeUtility.GetUnsafePtr(means.GetArray()), 0, UnsafeUtility.SizeOf<int>() * means.Length);

            for (var i = 0; i < data.DimensionX; i++)
            {
                var cluster = clusters[i];
                clusterItems[cluster] = clusterItems[cluster] + 1;
                for (var j = 0; j < data.DimensionY; j++)
                {
                    var val = means.Get(cluster, j);
                    means.Set(cluster, j, data.Get(i, j) + val);
                }
            }

            for (var k = 0; k < means.DimensionX; k++)
            {
                for (var a = 0; a < means.DimensionY; a++)
                {
                    var itemCount = clusterItems[k];
                    var value = means.Get(k, a);
                    value /= itemCount > 0 ? itemCount : 1;
                    means.Set(k, a, value);
                }
            }

            var totalDistance = 0.0f;
            var minDistances = new NativeArray<float>(clusterCount, Allocator.Temp, NativeArrayOptions.ClearMemory);
            for (var i = 0; i < clusterCount; ++i)
                minDistances[i] = float.MaxValue;

            for (var i = 0; i < data.DimensionX; i++)
            {
                var cluster = clusters[i];
                var distance = CalculateDistance(data, i, means, cluster);
                totalDistance += distance;
                if (distance < minDistances[cluster])
                {
                    minDistances[cluster] = distance;
                    centroids[cluster] = i;
                }
            }

            minDistances.Dispose();
            return totalDistance;
        }

        private static bool AssignClustering(MatrixMxN<float> data, NativeArray<int> clusters, ref NativeArray<int> centroidIdx, int clusterCount)
        {
            var changed = false;

            for (var i = 0; i < data.DimensionX; i++)
            {
                var minDistance = float.MaxValue;
                var minClusterIndex = -1;

                for (var k = 0; k < clusterCount; k++)
                {
                    var cd = centroidIdx[k];
                    var distance = CalculateDistance(data, i, data, cd);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        minClusterIndex = k;
                    }
                }

                if (minClusterIndex != -1 && clusters[i] != minClusterIndex)
                {
                    changed = true;
                    clusters[i] = minClusterIndex;
                }
            }

            return changed;
        }

        private unsafe static void ClusterInternal(MatrixMxN<float> data, NativeArray<int> clusters, MatrixMxN<float> means, NativeArray<int> centroids, NativeArray<int> clusterItems, int clusterCount, int maxIterations)
        {
            var hasChanges = true;
            var iteration = 0;

            var random = new Unity.Mathematics.Random(1);
            for (var i = 0; i < clusters.Length; ++i)
                clusters[i] = random.NextInt(0, clusterCount);
            while (hasChanges && iteration++ < maxIterations)
            {
                UnsafeUtility.MemSet(NativeArrayUnsafeUtility.GetUnsafePtr(clusterItems), 0, UnsafeUtility.SizeOf<int>() * clusterCount);
                CalculateClustering(data, clusters, ref means, ref centroids, clusterCount, ref clusterItems);
                hasChanges = AssignClustering(data, clusters, ref centroids, clusterCount);
            }
        }

        // Reference/Example function.
        // Ideally wrap this functionw with Burst. The following example is for Vector3 but should equally work well for Vector2 to VectorN
        public static int[] Cluster3(NativeArray<float3> items, int clusterCount, Allocator alloc, int maxIterations = 64)
        {
            var data = new MatrixMxN<float>(items.Length, 3, alloc, Unity.Collections.NativeArrayOptions.UninitializedMemory);
            var clusters = new NativeArray<int>(items.Length, alloc, NativeArrayOptions.UninitializedMemory);
            var means = new MatrixMxN<float>(clusterCount, 3, alloc, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < items.Length; i++)
            {
                data.Set(i, 0, items[i].x);
                data.Set(i, 1, items[i].y);
                data.Set(i, 2, items[i].z);
            }
            var centroids = new NativeArray<int>(clusterCount, alloc, NativeArrayOptions.UninitializedMemory);
            var clusterItems = new NativeArray<int>(clusterCount, alloc, NativeArrayOptions.UninitializedMemory);

            ClusterInternal(data, clusters, means, centroids, clusterItems, clusterCount, maxIterations);

            var returnData = centroids.ToArray();
            clusterItems.Dispose();
            centroids.Dispose();
            means.Dispose();
            clusters.Dispose();
            data.Dispose();
            return returnData;
        }

    }

}