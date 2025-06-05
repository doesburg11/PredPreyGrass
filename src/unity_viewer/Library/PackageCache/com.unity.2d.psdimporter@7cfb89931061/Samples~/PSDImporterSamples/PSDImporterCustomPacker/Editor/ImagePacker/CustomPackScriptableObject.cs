using System;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace PSDImporterCustomPacker
{
    [CreateAssetMenu(fileName = "Pipeline.asset", menuName = "2D/Custom PSDImporter Pipeline")]
    class CustomPackScriptableObject : ScriptableObject
    {
        [SerializeField]
        /// Objects to applied the ScriptableObject to.
        Object[] m_ObjectsToApply;
        [SerializeField]
        /// Objects that the ScriptableObject was applied to previously.
        Object[] m_ObjectsAppliedPreviously;

        /// <summary>
        /// Method that will be called by the PSDImporter to pack the images.
        /// </summary>
        /// <param name="buffers">Image buffer of each layer in the Photoshop file.</param>
        /// <param name="width">The width of each layer in the Photoshop file.</param>
        /// <param name="height">The height of each layer in the Photoshop file.</param>
        /// <param name="padding">Padding value specified by user in the PSDImporter inspector. The padding value is used to create gaps between each layer that is packed in the final texture.</param>
        /// <param name="spriteSizeExpand">Sprite size expansion value specified by the user in the PSDImporter inspector. The value is to override the size of the Sprite size.</param>
        /// <param name="outPackedBuffer">Image buffer of the packed image.</param>
        /// <param name="outPackedBufferWidth">Packed image buffer width.</param>
        /// <param name="outPackedBufferHeight">Packed image buffer height.</param>
        /// <param name="outPackedRect">Position of each layer in the packed image.</param>
        /// <param name="outUVTransform">Transform value of each layer from the Photoshop file to the packed image.</param>
        /// <param name="requireSquarePOT">Determine if the packed image needs to be in Power of Two format.</param>
        void PackImage(NativeArray<Color32>[] buffers, int[] width, int[] height, int padding, uint spriteSizeExpand, out NativeArray<Color32> outPackedBuffer, out int outPackedBufferWidth, out int outPackedBufferHeight, out RectInt[] outPackedRect, out Vector2Int[] outUVTransform, bool requireSquarePOT = false)
        {
            // The packing algorithm is an example and extracted from the packing algorithm used in PSDImporter 6.0.2
            ImagePacker.Pack(buffers, width, height, padding, out outPackedBuffer, out outPackedBufferWidth, out outPackedBufferHeight, out outPackedRect, out outUVTransform);
        }

        /// <summary>
        /// Utility to assign the ScriptableObject to the m_Pipeline property value of the PSDImporter.
        /// </summary>
        public void ApplyPipeline()
        {
            if (m_ObjectsAppliedPreviously != null)
            {
                for (int i = 0; i < m_ObjectsAppliedPreviously.Length; ++i)
                {
                    if (m_ObjectsAppliedPreviously[i] == null)
                        continue;
                    var im = AssetImporter.GetAtPath(AssetDatabase.GetAssetPath(m_ObjectsAppliedPreviously[i]));
                    if (im != null)
                    {
                        var so = new SerializedObject(im);
                        var pipeline = so.FindProperty("m_Pipeline");
                        var pipelineVersion = so.FindProperty("m_PipelineVersion");
                        if (pipeline != null && pipeline.objectReferenceValue == this && pipelineVersion != null)
                        {
                            pipeline.objectReferenceValue = null;
                            // change the pipeline version number value so we don't get importer import inconsistency
                            pipelineVersion.stringValue = DateTime.Now.ToString();
                            so.ApplyModifiedProperties();
                            im.SaveAndReimport();
                        }
                    }
                }

                m_ObjectsAppliedPreviously = null;
            }

            if (m_ObjectsToApply != null)
            {
                m_ObjectsAppliedPreviously = new Object[m_ObjectsToApply.Length];
                for(int i = 0; i < m_ObjectsToApply.Length; i++)
                {
                    var im = AssetImporter.GetAtPath(AssetDatabase.GetAssetPath(m_ObjectsToApply[i]));
                    if (im != null)
                    {
                        var so = new SerializedObject(im);
                        var pipeline = so.FindProperty("m_Pipeline");
                        var pipelineVersion = so.FindProperty("m_PipelineVersion");
                        if (pipeline != null && pipeline.objectReferenceValue != this && pipelineVersion != null)
                        {
                            pipeline.objectReferenceValue = this;
                            // change the pipeline version number value so we don't get importer import inconsistency
                            pipelineVersion.stringValue = DateTime.Now.ToString();
                            so.ApplyModifiedProperties();
                            im.SaveAndReimport();
                            m_ObjectsAppliedPreviously[i] = m_ObjectsToApply[i];
                        }
                    }
                }
            }
        }
    }
}