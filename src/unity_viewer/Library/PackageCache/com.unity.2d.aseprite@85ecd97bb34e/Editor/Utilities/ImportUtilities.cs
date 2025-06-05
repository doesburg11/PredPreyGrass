using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor.Animations;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal class UniqueNameGenerator
    {
        readonly Dictionary<int, HashSet<int>> m_NameHashes = new();

        public string GetUniqueName(string name, int parentIndex = -1, bool logNewNameGenerated = false, UnityEngine.Object context = null)
        {
            if (!m_NameHashes.ContainsKey(parentIndex))
                m_NameHashes.Add(parentIndex, new HashSet<int>());
            var nameHashes = m_NameHashes[parentIndex];
            return GetUniqueName(name, nameHashes, logNewNameGenerated, context);
        }

        static string GetUniqueName(string name, HashSet<int> stringHash, bool logNewNameGenerated = false, UnityEngine.Object context = null)
        {
            var sanitizedName = string.Copy(SanitizeName(name));
            string uniqueName = sanitizedName;
            int index = 1;
            while (true)
            {
                var hash = GetStringHash(uniqueName);
                if (!stringHash.Contains(hash))
                {
                    stringHash.Add(hash);
                    if (logNewNameGenerated && sanitizedName != uniqueName)
                        Debug.Log($"Asset name {name} is changed to {uniqueName} to ensure uniqueness", context);
                    return uniqueName;
                }
                uniqueName = $"{sanitizedName}_{index}";
                ++index;
            }
        }

        static string SanitizeName(string name)
        {
            name = name.Replace('\0', ' ');
            string newName = null;
            // We can't create asset name with these name.
            if ((name.Length == 2 && name[0] == '.' && name[1] == '.')
                || (name.Length == 1 && name[0] == '.')
                || (name.Length == 1 && name[0] == '/'))
                newName += name + "_";

            if (!string.IsNullOrEmpty(newName))
            {
                Debug.LogWarning($"File contains layer with invalid name for generating asset. {name} is renamed to {newName}");
                return newName;
            }
            return name;
        }

        static int GetStringHash(string str)
        {
            var md5Hasher = MD5.Create();
            var hashed = md5Hasher.ComputeHash(Encoding.UTF8.GetBytes(str));
            return BitConverter.ToInt32(hashed, 0);
        }
    }

    [BurstCompile]
    internal static class ImportUtilities
    {
        public static void SaveAllPalettesToDisk(AsepriteFile file)
        {
            for (var i = 0; i < file.frameData.Count; ++i)
            {
                var frame = file.frameData[i];
                for (var m = 0; m < frame.chunkCount; ++m)
                {
                    var chunk = frame.chunks[m];
                    if (chunk.chunkType == ChunkTypes.Palette)
                        PaletteToDisk(chunk as PaletteChunk);
                }
            }
        }

        static void PaletteToDisk(PaletteChunk palette)
        {
            var noOfEntries = palette.noOfEntries;
            const int cellSize = 32;
            const int columns = 3;
            var rows = Mathf.CeilToInt(noOfEntries / (float)3);

            const int width = columns * cellSize;
            var height = rows * cellSize;
            var buffer = new Color32[width * height];

            for (var i = 0; i < buffer.Length; ++i)
            {
                var x = i % width;
                var y = i / width;

                var imgColumn = x / 32;
                var imgRow = y / 32;
                var paletteEntry = imgColumn + (imgRow * columns);
                if (paletteEntry < palette.noOfEntries)
                    buffer[i] = palette.entries[paletteEntry].color;
            }

            SaveToPng(buffer, width, height);
        }

        public static string SaveToPng(NativeArray<Color32> buffer, int width, int height)
        {
            return SaveToPng(buffer.ToArray(), width, height);
        }

        static string SaveToPng(Color32[] buffer, int width, int height)
        {
            if (width == 0 || height == 0)
                return "No .png generated.";

            var texture2D = new Texture2D(width, height);
            texture2D.SetPixels32(buffer);
            var png = texture2D.EncodeToPNG();
            var path = Application.dataPath + $"/tex_{System.Guid.NewGuid().ToString()}.png";
            var fileStream = System.IO.File.Create(path);
            fileStream.Write(png);
            fileStream.Close();

            UnityEngine.Object.DestroyImmediate(texture2D);

            return path;
        }

        public static void ExportAnimationAssets(AsepriteImporter[] importers, bool exportClips, bool exportController)
        {
            var savePath = EditorUtility.SaveFolderPanel(
                "Export Animation Assets",
                Application.dataPath, "");

            ExportAnimationAssets(savePath, importers, exportClips, exportController);
        }

        public static void ExportAnimationAssets(string savePath, AsepriteImporter[] importers, bool exportClips, bool exportController)
        {
            if (string.IsNullOrEmpty(savePath))
                return;

            for (var i = 0; i < importers.Length; ++i)
            {
                var importedObjectPath = importers[i].assetPath;
                AnimationClip[] clips;

                if (exportClips)
                    clips = ExportAnimationClips(importedObjectPath, savePath);
                else
                    clips = GetAllClipsFromController(importedObjectPath);

                if (exportController)
                    ExportAnimatorController(importers[i], clips, savePath);
            }
        }

        static AnimationClip[] ExportAnimationClips(string importedObjectPath, string path)
        {
            var relativePath = FileUtil.GetProjectRelativePath(path);
            var animationClips = GetAllClipsFromController(importedObjectPath);

            var clips = new List<AnimationClip>();
            for (var i = 0; i < animationClips.Length; ++i)
            {
                var clip = animationClips[i];
                var clipPath = $"{relativePath}/{clip.name}.anim";
                var result = AssetDatabase.ExtractAsset(clip, clipPath);
                if (!string.IsNullOrEmpty(result))
                    Debug.LogWarning(result);

                var newClip = AssetDatabase.LoadAssetAtPath<AnimationClip>(clipPath);
                clips.Add(newClip);
            }
            return clips.ToArray();
        }

        static AnimationClip[] GetAllClipsFromController(string assetPath)
        {
            var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(assetPath);
            return controller.animationClips;
        }

        static void ExportAnimatorController(AsepriteImporter importer, AnimationClip[] clips, string path)
        {
            var relativePath = FileUtil.GetProjectRelativePath(path);

            var importedObjectPath = importer.assetPath;
            var fileName = System.IO.Path.GetFileNameWithoutExtension(importedObjectPath);

            var controllerPath = $"{relativePath}/{fileName}.controller";
            var controller = AnimatorController.CreateAnimatorControllerAtPath(controllerPath);

            for (var i = 0; i < clips.Length; ++i)
                controller.AddMotion(clips[i]);
        }
        
        public static float2 CalculateCellPivot(RectInt cellRect, uint spritePadding, int2 canvasSize, SpriteAlignment alignment, float2 customPivot)
        {
            if (cellRect.width == 0 || cellRect.height == 0)
                return float2.zero;

            var scaleX = canvasSize.x / (float)cellRect.width;
            var scaleY = canvasSize.y / (float)cellRect.height;
            var halfSpritePadding = spritePadding / 2f;

            var pivot = new float2((cellRect.x - halfSpritePadding) / (float)canvasSize.x, (cellRect.y - halfSpritePadding) / (float)canvasSize.y);
            pivot *= -1f;

            float2 alignmentPos;
            if (alignment == SpriteAlignment.Custom)
                alignmentPos = customPivot;
            else
                alignmentPos = PivotAlignmentToVector(alignment);

            pivot.x += alignmentPos.x;
            pivot.y += alignmentPos.y;

            pivot.x *= scaleX;
            pivot.y *= scaleY;

            return pivot;
        }

        public static float2 PivotAlignmentToVector(SpriteAlignment alignment)
        {
            switch (alignment)
            {
                case SpriteAlignment.Center:
                    return new float2(0.5f, 0.5f);
                case SpriteAlignment.TopLeft:
                    return new float2(0f, 1f);
                case SpriteAlignment.TopCenter:
                    return new float2(0.5f, 1f);
                case SpriteAlignment.TopRight:
                    return new float2(1f, 1f);
                case SpriteAlignment.LeftCenter:
                    return new float2(0f, 0.5f);
                case SpriteAlignment.RightCenter:
                    return new float2(1f, 0.5f);
                case SpriteAlignment.BottomLeft:
                    return new float2(0f, 0f);
                case SpriteAlignment.BottomCenter:
                    return new float2(0.5f, 0f);
                case SpriteAlignment.BottomRight:
                    return new float2(1f, 0f);
                default:
                    return new float2(0f, 0f);
            }
        }

        public static string GetCellName(string baseName, int frameIndex, int noOfFrames, bool isMerged)
        {
            if (noOfFrames == 1)
                return baseName;
            return isMerged ? $"Frame_{frameIndex}" : $"{baseName}_Frame_{frameIndex}";
        }

        public static void DisposeIfCreated<T>(this NativeArray<T> arr) where T : struct
        {
            if (arr == default || !arr.IsCreated)
                return;
            var handle = NativeArrayUnsafeUtility.GetAtomicSafetyHandle(arr);
            if (!AtomicSafetyHandle.IsHandleValid(handle))
                return;

            arr.Dispose();
        }

        public static bool IsLayerVisible(int layerIndex, in List<Layer> layers)
        {
            var layer = layers[layerIndex];
            var isVisible = (layer.layerFlags & LayerFlags.Visible) != 0;
            if (!isVisible)
                return false;

            if (layer.parentIndex != -1)
                isVisible = IsLayerVisible(layer.parentIndex, in layers);
            return isVisible;
        }
        
        [BurstCompile]
        public static bool IsEmptyImage(in NativeArray<Color32> image)
        {
            for (var i = 0; i < image.Length; ++i)
            {
                if (image[i].a > 0)
                    return false;
            }
            return true;
        }
    }
}
