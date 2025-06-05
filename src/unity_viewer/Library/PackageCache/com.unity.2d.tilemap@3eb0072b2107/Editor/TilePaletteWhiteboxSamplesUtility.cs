using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEditor.PackageManager.Requests;
using UnityEditor.PackageManager.UI;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal static class TilePaletteWhiteboxSamplesUtility
    {
        private static readonly String whiteboxFilterText = "Whitebox";
        private static readonly List<String> m_WhiteboxSamplesPackages = new List<String>(new[] {"com.unity.2d.tilemap", "com.unity.2d.tilemap.extras"});

        private static ListRequest m_WhiteboxPackageRequest;
        private static List<Sample> m_WhiteboxSamples;
        private static List<String> m_WhiteboxSampleNames;
        private static String[] m_WhiteboxSampleNamesArray;
        private static int m_LongestNameIndex = -1;

        public static List<Sample> whiteboxSamples
        {
            get
            {
                if (m_WhiteboxSamples == null)
                    GetWhiteboxSamples();
                return m_WhiteboxSamples;
            }
        }

        public static List<String> whiteboxSampleNames
        {
            get
            {
                if (m_WhiteboxSampleNames == null)
                    GetWhiteboxSamples();
                return m_WhiteboxSampleNames;
            }
        }

        public static String[] whiteboxSampleNamesArray
        {
            get
            {
                if (m_WhiteboxSampleNamesArray == null)
                    GetWhiteboxSamples();
                return m_WhiteboxSampleNamesArray;
            }
        }

        public static String longestName
        {
            get
            {
                if (m_LongestNameIndex >= 0 && m_LongestNameIndex < m_WhiteboxSampleNames.Count)
                    return m_WhiteboxSampleNames[m_LongestNameIndex];
                return null;
            }
        }

        private static void GetWhiteboxSamples()
        {
            m_WhiteboxSamples = new List<Sample>();
            m_WhiteboxSampleNames = new List<String>();
            m_LongestNameIndex = -1;

            var packages = PackageManager.PackageInfo.GetAllRegisteredPackages();
            foreach (var packageInfo in packages)
            {
                if (!m_WhiteboxSamplesPackages.Contains(packageInfo.name))
                    continue;

                var samples = Sample.FindByPackage(packageInfo.name, packageInfo.version);
                foreach (var sample in samples)
                {
                    if (!sample.displayName.Contains(whiteboxFilterText, StringComparison.CurrentCultureIgnoreCase))
                        continue;

                    m_WhiteboxSamples.Add(sample);
                    m_WhiteboxSampleNames.Add(sample.displayName);

                    if (m_LongestNameIndex == -1 || longestName.Length < sample.displayName.Length)
                        m_LongestNameIndex = m_WhiteboxSampleNames.Count - 1;
                }
            }
            m_WhiteboxSampleNamesArray = m_WhiteboxSampleNames.ToArray();
        }

        internal static void ImportWhiteboxSample(int index)
        {
            if (index >= whiteboxSamples.Count)
                return;

            var sample = whiteboxSamples[index];
            ImportWhiteboxSample(sample);
        }

        internal static void ImportWhiteboxSample(Sample sample)
        {
            sample.Import(Sample.ImportOptions.HideImportWindow);

            GridPalettes.CleanCache();

            // Try to select the new palette as the current palette
            var directoryInfo = new DirectoryInfo(sample.importPath);
            var fileInfos = directoryInfo.GetFiles("*.prefab");
            foreach (var file in fileInfos)
            {
                var dataPath = FileUtil.NiceWinPath(Application.dataPath);
                var absolutePath = FileUtil.NiceWinPath(file.FullName);
                if (!absolutePath.StartsWith(dataPath))
                    return;

                var relativePath= FileUtil.CombinePaths("Assets", absolutePath.Substring(dataPath.Length));
                var gridPalette = AssetDatabase.LoadAssetAtPath(relativePath, typeof(GridPalette)) as GridPalette;
                if (gridPalette != null)
                {
                    var go = AssetDatabase.LoadAssetAtPath(relativePath, typeof(GameObject)) as GameObject;
                    GridPaintingState.palette = go;
                    return;
                }
            }
        }

        internal static void DuplicateWhiteboxSample(int index)
        {
            if (index >= whiteboxSamples.Count)
                return;

            var sample = whiteboxSamples[index];
            DuplicateWhiteboxSample(sample);
        }

        internal static void DuplicateWhiteboxSample(Sample sample)
        {
            var path = ProjectBrowser.s_LastInteractedProjectBrowser != null
                ? ProjectBrowser.s_LastInteractedProjectBrowser.GetActiveFolderPath()
                : "Assets";

            path = EditorUtility.SaveFilePanelInProject("Generate Whitebox Palette Asset", sample.displayName, "prefab",
                "Generate Whitebox Palette Asset", path);
            if (String.IsNullOrWhiteSpace(path))
                return;

            var dirPath = FileUtil.UnityGetDirectoryName(path);
            var fileName = FileUtil.UnityGetFileNameWithoutExtension(path);
            DuplicateWhiteboxSample(sample, dirPath, fileName);
        }

        private static void ReplaceGuidInMetaFile(FileInfo fileInfo, out string oldGuid, out string newGuid)
        {
            const string kGuidLine = "guid: ";

            var reader = fileInfo.OpenText();
            reader.ReadLine();
            var guidLine = reader.ReadLine();
            if (!guidLine.StartsWith(kGuidLine))
            {
                reader.Close();
                throw new FormatException("Meta File does not have a valid GUID.");
            }

            oldGuid = guidLine.Remove(0, kGuidLine.Length);
            oldGuid.TrimEnd(Environment.NewLine.ToCharArray());

            newGuid = GUID.Generate().ToString();
            reader.Close();

            var absolutePath = FileUtil.NiceWinPath(fileInfo.FullName);
            var readAllText = File.ReadAllText(absolutePath);
            var replace = readAllText.Replace(oldGuid, newGuid);
            File.WriteAllText(absolutePath, replace);
        }

        internal static void DuplicateWhiteboxSample(Sample sample, string path, string paletteName = null)
        {
            AssetDatabase.DisallowAutoRefresh();
            AssetDatabase.StartAssetEditing();

            var sampleDirectoryInfo = new DirectoryInfo(sample.resolvedPath);
            var sampleMetaFileInfos = sampleDirectoryInfo.GetFiles("*.meta");
            var sampleMetaNames = new HashSet<string>();
            foreach (var sampleMetaFileInfo in sampleMetaFileInfos)
                sampleMetaNames.Add(sampleMetaFileInfo.Name);

            var tempPath = AssetDatabase.GetUniquePathNameAtSelectedPath("Temp");
            FileUtil.CreateOrCleanDirectory(tempPath);
            FileUtil.CopyDirectoryRecursiveFiltered(sample.resolvedPath, tempPath, true, @"\.sample\.json$");

            // Fix meta references for copied files
            var textureIdMap = new Dictionary<string, string>();
            var textureDirectoryInfo = new DirectoryInfo(FileUtil.CombinePaths(tempPath, "Textures"));
            var textureMetaFileInfos = textureDirectoryInfo.GetFiles("*.meta");
            foreach (var metaFileInfo in textureMetaFileInfos)
            {
                ReplaceGuidInMetaFile(metaFileInfo, out var oldGuid, out var newGuid);
                textureIdMap.Add(oldGuid, newGuid);
            }

            var tileIdMap = new Dictionary<string, string>();
            var tileDirectoryInfo = new DirectoryInfo(FileUtil.CombinePaths(tempPath, "Tiles"));
            var tileMetaFileInfos = tileDirectoryInfo.GetFiles("*.meta");
            foreach (var metaFileInfo in tileMetaFileInfos)
            {
                ReplaceGuidInMetaFile(metaFileInfo, out var oldGuid, out var newGuid);
                tileIdMap.Add(oldGuid, newGuid);
            }

            var tempBaseDirectoryInfo = new DirectoryInfo(tempPath);
            var baseMetaFileInfos = tempBaseDirectoryInfo.GetFiles("*.meta");
            foreach (var metaFileInfo in baseMetaFileInfos)
            {
                if (sampleMetaNames.Contains(metaFileInfo.Name))
                    ReplaceGuidInMetaFile(metaFileInfo, out var oldGuid, out var newGuid);
            }

            var tileAssetFileInfos = tileDirectoryInfo.GetFiles("*.asset");
            var textureIdReplaceArray = new string[textureIdMap.Count * 2];
            {
                var i = 0;
                foreach (var pair in textureIdMap)
                {
                    textureIdReplaceArray[i++] = pair.Key;
                    textureIdReplaceArray[i++] = pair.Value;
                }
            }
            foreach (var tileAssetFileInfo in tileAssetFileInfos)
            {
                var absolutePath = FileUtil.NiceWinPath(tileAssetFileInfo.FullName);
                FileUtil.ReplaceTextRegex(absolutePath, textureIdReplaceArray);
            }

            var prefabFileInfos = tempBaseDirectoryInfo.GetFiles("*.prefab");
            var tileIdReplaceArray = new string[tileIdMap.Count * 2];
            {
                var i = 0;
                foreach (var pair in tileIdMap)
                {
                    tileIdReplaceArray[i++] = pair.Key;
                    tileIdReplaceArray[i++] = pair.Value;
                }
            }
            foreach (var prefabFileInfo in prefabFileInfos)
            {
                var prefabPath = FileUtil.NiceWinPath(prefabFileInfo.FullName);
                FileUtil.ReplaceTextRegex(prefabPath, tileIdReplaceArray);

                // Rename Palette if user has changed the name
                if (!String.IsNullOrEmpty(paletteName))
                {
                    var paletteNameWithExtension = $"{paletteName}.prefab";
                    if (!String.Equals(paletteNameWithExtension, prefabFileInfo.Name))
                    {
                        var prefabName = FileUtil.CombinePaths(prefabFileInfo.DirectoryName, paletteNameWithExtension);
                        FileUtil.MoveFileIfExists(prefabPath, prefabName);
                        FileUtil.MoveFileIfExists($"{prefabPath}.meta", FileUtil.CombinePaths(prefabFileInfo.DirectoryName, $"{paletteNameWithExtension}.meta"));
                    }
                }
            }

            FileUtil.CopyDirectoryRecursive(tempPath, path, true);
            FileUtil.DeleteFileOrDirectory(tempPath);

            AssetDatabase.StopAssetEditing();
            AssetDatabase.AllowAutoRefresh();

            AssetDatabase.Refresh();

            GridPalettes.CleanCache();

            // Try to select the new palette as the current palette
            var baseDirectoryInfo = new DirectoryInfo(path);
            prefabFileInfos = baseDirectoryInfo.GetFiles("*.prefab");
            foreach (var prefabFileInfo in prefabFileInfos)
            {
                var dataPath = FileUtil.NiceWinPath(Application.dataPath);
                var absolutePath = FileUtil.NiceWinPath(prefabFileInfo.FullName);
                if (!absolutePath.StartsWith(dataPath))
                    return;

                var relativePath= FileUtil.CombinePaths("Assets", absolutePath.Substring(dataPath.Length));
                var gridPalette = AssetDatabase.LoadAssetAtPath(relativePath, typeof(GridPalette)) as GridPalette;
                if (gridPalette != null)
                {
                    var go = AssetDatabase.LoadAssetAtPath(relativePath, typeof(GameObject)) as GameObject;
                    GridPaintingState.palette = go;
                    return;
                }
            }
        }
    }
}
