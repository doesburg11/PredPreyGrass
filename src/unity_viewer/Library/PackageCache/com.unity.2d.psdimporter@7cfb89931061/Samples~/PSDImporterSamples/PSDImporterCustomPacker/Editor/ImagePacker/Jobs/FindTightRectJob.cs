using System;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace PSDImporterCustomPacker
{
    internal struct FindTightRectJob : IJobParallelFor
    {
        [ReadOnly, DeallocateOnJobCompletion]
        NativeArray<IntPtr> m_Buffers;
        [ReadOnly, DeallocateOnJobCompletion]
        NativeArray<int> m_Width;
        [ReadOnly, DeallocateOnJobCompletion]
        NativeArray<int> m_Height;
        NativeArray<RectInt> m_Output;

        public unsafe void Execute(int index)
        {
            var rect = new RectInt(m_Width[index], m_Height[index], 0, 0);

            if (m_Height[index] == 0 || m_Width[index] == 0)
            {
                m_Output[index] = rect;
                return;
            }

            var color = (Color32*)m_Buffers[index].ToPointer();
            for (int i = 0; i < m_Height[index]; ++i)
            {
                for (int j = 0; j < m_Width[index]; ++j)
                {
                    if (color->a != 0)
                    {
                        rect.x = Mathf.Min(j, rect.x);
                        rect.y = Mathf.Min(i, rect.y);
                        rect.width = Mathf.Max(j, rect.width);
                        rect.height = Mathf.Max(i, rect.height);
                    }
                    ++color;
                }
            }
            rect.width = Mathf.Max(0, rect.width - rect.x + 1);
            rect.height = Mathf.Max(0, rect.height - rect.y + 1);
            m_Output[index] = rect;
        }

        public static unsafe RectInt[] Execute(NativeArray<Color32>[] buffers, int[] width, int[] height)
        {
            var job = new FindTightRectJob();
            job.m_Buffers = new NativeArray<IntPtr>(buffers.Length, Allocator.TempJob);
            job.m_Width = new NativeArray<int>(width.Length, Allocator.TempJob);
            job.m_Height = new NativeArray<int>(height.Length, Allocator.TempJob);

            for (var i = 0; i < buffers.Length; ++i)
            {
                job.m_Buffers[i] = new IntPtr(buffers[i].GetUnsafeReadOnlyPtr());
                job.m_Width[i] = width[i];
                job.m_Height[i] = height[i];
            }

            job.m_Output = new NativeArray<RectInt>(buffers.Length, Allocator.TempJob);

            // Ensure all jobs are completed before we return since we don't own the buffers
            job.Schedule(buffers.Length, 1).Complete();
            var rects = job.m_Output.ToArray();
            job.m_Output.Dispose();
            return rects;
        }
    }
}