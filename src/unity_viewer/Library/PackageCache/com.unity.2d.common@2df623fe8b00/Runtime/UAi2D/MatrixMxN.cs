using System;
using Unity.Collections;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace UnityEngine.U2D.Common.UAi
{
    /// <summary>
    /// Matrix (MxN). Used within UAi and constrained to
    /// 1. No Resize.
    /// 2. Only be used within the created thread. Read 1.
    /// 3. Read/Write access are all fast-paths.
    /// 4. Generalized to handle any dimension Matrix MxN.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [StructLayout(LayoutKind.Sequential)]
    [DebuggerDisplay("Length = {Length}")]
    [DebuggerTypeProxy(typeof(MatrixMxNDebugView<>))]
    internal unsafe struct MatrixMxN<T> : IDisposable where T : struct
    {
        internal NativeArray<T> m_Array;
        internal int m_Width;
        internal int m_Height;
        internal Allocator m_AllocLabel;
        internal NativeArrayOptions m_Options;

        public MatrixMxN(int width, int height, Allocator allocMode, NativeArrayOptions options)
        {
            m_Width = width;
            m_Height = height;
            m_Array = new NativeArray<T>(m_Width * m_Height, allocMode, options);
            m_AllocLabel = allocMode;
            m_Options = options;
        }

        unsafe T this[int index]
        {
            get
            {
                return m_Array[index];
            }
            set
            {
                m_Array[index] = value;
            }
        }

        public NativeArray<T> GetArray()
        {
            return m_Array;
        }

        public T Get(int x, int y)
        {
            return m_Array[(x * m_Height) + y];
        }

        public void Set(int x, int y, T v)
        {
            m_Array[(x * m_Height) + y] = v;
        }

        public bool IsCreated => m_Array.IsCreated;

        public int Length => m_Width * m_Height;

        public int DimensionX => m_Width;

        public int DimensionY => m_Height;

        public void Dispose()
        {
            m_Array.Dispose();
            m_Width = 0;
            m_Height = 0;
        }

        // Should only ever be used for Debugging.
        public void CopyTo(T[] array)
        {
            m_Array.CopyTo(array);
        }
    }

    /// <summary>
    /// DebuggerTypeProxy for <see cref="Array{T}"/>
    /// </summary>
    internal sealed class MatrixMxNDebugView<T> where T : struct
    {
        private MatrixMxN<T> array;

        public MatrixMxNDebugView(MatrixMxN<T> array)
        {
            this.array = array;
        }

        public T[] Items
        {
            get
            {
                var ret = new T[array.Length];
                array.CopyTo(ret);
                return ret;
            }
        }
    }

}