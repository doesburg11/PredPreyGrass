using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using PDNWrapper;
using Unity.Collections;
using UnityEngine;

#if ENABLE_2D_ANIMATION
using UnityEditor.U2D.Animation;
#endif

namespace UnityEditor.U2D.PSD
{
    class UniqueNameGenerator
    {
        HashSet<int> m_NameHash = new HashSet<int>();

        public bool ContainHash(int i)
        {
            return m_NameHash.Contains(i);
        }

        public void AddHash(int i)
        {
            m_NameHash.Add(i);
        }

        public void AddHash(string name)
        {
            m_NameHash.Add(GetStringHash(name));
        }
        
        public string GetUniqueName(string name, bool logNewNameGenerated = false, UnityEngine.Object context = null)
        {
            return GetUniqueName(name, m_NameHash);
        }
        
        static string GetUniqueName(string name, HashSet<int> stringHash, bool logNewNameGenerated = false, UnityEngine.Object context = null)
        {
            var sanitizedName = string.Copy(SanitizeName(name));
            var uniqueName = sanitizedName;
            var index = 1;
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

        static int GetStringHash(string str)
        {
            var md5Hasher = MD5.Create();
            var hashed = md5Hasher.ComputeHash(Encoding.UTF8.GetBytes(str));
            return BitConverter.ToInt32(hashed, 0);
        }
        
        public static string SanitizeName(string name)
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
    }
        
    class GameObjectCreationFactory : UniqueNameGenerator
    {
        public GameObjectCreationFactory(IList<string> names)
        {
            if (names != null)
            {
                foreach (var name in names)
                    GetUniqueName(name);
            }
        }
        
        public GameObject CreateGameObject(string name, params System.Type[] components)
        {
            var newName = GetUniqueName(name);
            return new GameObject(newName, components);
        }
    }

    internal static class ImportUtilities
    {
        public static string SaveToPng(NativeArray<Color32> buffer, int width, int height)
        {
            if (!buffer.IsCreated ||
                buffer.Length == 0 ||
                width == 0 ||
                height == 0)
                return "No .png generated.";
            
            var texture2D = new Texture2D(width, height);
            texture2D.SetPixels32(buffer.ToArray());
            var png = texture2D.EncodeToPNG();
            var path = Application.dataPath + $"/tex_{System.Guid.NewGuid().ToString()}.png";
            var fileStream = System.IO.File.Create(path);
            fileStream.Write(png);
            fileStream.Close();
            
            UnityEngine.Object.DestroyImmediate(texture2D);

            return path;
        }
        
        public static void ValidatePSDLayerId(IEnumerable<PSDLayer> oldPsdLayer, IEnumerable<BitmapLayer> layers, UniqueNameGenerator uniqueNameGenerator)
        {
            // first check if all layers are unique. If not, we use back the previous layer id based on name match
            var uniqueIdSet = new HashSet<int>();
            var useOldID = false;
            foreach(var layer in layers)
            {
                if (uniqueIdSet.Contains(layer.LayerID))
                {
                    useOldID = true;
                    break;   
                }
                uniqueIdSet.Add(layer.LayerID);
            }

            for (var i = 0; i < layers.Count(); ++i)
            {
                var childBitmapLayer = layers.ElementAt(i);
                // fix case 1291323
                if (useOldID)
                {
                    var oldLayers = oldPsdLayer.Where(x => x.name == childBitmapLayer.Name);
                    if (oldLayers.Count() == 0)
                        oldLayers = oldPsdLayer.Where(x => x.layerID == childBitmapLayer.Name.GetHashCode()); 
                    // pick one that is not already on the list
                    foreach (var ol in oldLayers)
                    {
                        if (!uniqueNameGenerator.ContainHash(ol.layerID))
                        {
                            childBitmapLayer.LayerID = ol.layerID;
                            break;
                        }
                    }
                }
            
                if (uniqueNameGenerator.ContainHash(childBitmapLayer.LayerID))
                {
                    var layerName = UniqueNameGenerator.SanitizeName(childBitmapLayer.Name);
                    var importWarning = $"Layer {layerName}: LayerId is not unique. Mapping will be done by Layer's name.";
                    layerName = uniqueNameGenerator.GetUniqueName(layerName);
                    if (layerName != childBitmapLayer.Name)
                        importWarning += "\nLayer names are not unique. Please ensure they are unique to for SpriteRect to be mapped back correctly.";
                    childBitmapLayer.LayerID = layerName.GetHashCode();
                    Debug.LogWarning(importWarning);
                }
                else
                    uniqueNameGenerator.AddHash(childBitmapLayer.LayerID);
                if (childBitmapLayer.ChildLayer != null)
                {
                    ValidatePSDLayerId(oldPsdLayer, childBitmapLayer.ChildLayer, uniqueNameGenerator);
                }
            }
        }        
        
        public static void TranslatePivotPoint(Vector2 pivot, Rect rect, out SpriteAlignment alignment, out Vector2 customPivot)
        {
            customPivot = pivot;
            if (new Vector2(rect.xMin, rect.yMax) == pivot)
                alignment = SpriteAlignment.TopLeft;
            else if(new Vector2(rect.center.x, rect.yMax) == pivot)
                alignment = SpriteAlignment.TopCenter;
            else if(new Vector2(rect.xMax, rect.yMax) == pivot)
                alignment = SpriteAlignment.TopRight;
            else if(new Vector2(rect.xMin, rect.center.y) == pivot)
                alignment = SpriteAlignment.LeftCenter;
            else if(new Vector2(rect.center.x, rect.center.y) == pivot)
                alignment = SpriteAlignment.Center;
            else if(new Vector2(rect.xMax, rect.center.y) == pivot)
                alignment = SpriteAlignment.RightCenter;
            else if(new Vector2(rect.xMin, rect.yMin) == pivot)
                alignment = SpriteAlignment.BottomLeft;
            else if(new Vector2(rect.center.x, rect.yMin) == pivot)
                alignment = SpriteAlignment.BottomCenter;
            else if(new Vector2(rect.xMax, rect.yMin) == pivot)
                alignment = SpriteAlignment.BottomRight;
            else
                alignment = SpriteAlignment.Custom;
        }   
        
        public static Vector2 GetPivotPoint(Rect rect, SpriteAlignment alignment, Vector2 customPivot)
        {
            switch (alignment)
            {
                case SpriteAlignment.TopLeft:
                    return new Vector2(rect.xMin, rect.yMax);

                case SpriteAlignment.TopCenter:
                    return new Vector2(rect.center.x, rect.yMax);

                case SpriteAlignment.TopRight:
                    return new Vector2(rect.xMax, rect.yMax);

                case SpriteAlignment.LeftCenter:
                    return new Vector2(rect.xMin, rect.center.y);

                case SpriteAlignment.Center:
                    return new Vector2(rect.center.x, rect.center.y);

                case SpriteAlignment.RightCenter:
                    return new Vector2(rect.xMax, rect.center.y);

                case SpriteAlignment.BottomLeft:
                    return new Vector2(rect.xMin, rect.yMin);

                case SpriteAlignment.BottomCenter:
                    return new Vector2(rect.center.x, rect.yMin);

                case SpriteAlignment.BottomRight:
                    return new Vector2(rect.xMax, rect.yMin);

                case SpriteAlignment.Custom:
                    return new Vector2(customPivot.x * rect.width, customPivot.y * rect.height);
            }
            return Vector2.zero;
        }     
        
        public static string GetUniqueSpriteName(string name, UniqueNameGenerator generator, bool keepDupilcateSpriteName)
        {
            if (keepDupilcateSpriteName)
                return name;
            return generator.GetUniqueName(name);
        }
        
        public static bool VisibleInHierarchy(List<PSDLayer> psdGroup, int index)
        {
            var psdLayer = psdGroup[index];
            var parentVisible = true;
            if (psdLayer.parentIndex >= 0)
                parentVisible = VisibleInHierarchy(psdGroup, psdLayer.parentIndex);
            return parentVisible && psdLayer.isVisible;
        }        
        
        public static bool IsSpriteMetaDataDefault(SpriteMetaData metaData)
        {
            return metaData.spriteID == default ||
                   metaData.rect == Rect.zero;
        }

#if ENABLE_2D_ANIMATION        
        public static bool SpriteIsMainFromSpriteLib(List<SpriteCategory> categories, string spriteId, out string categoryName)
        {
            categoryName = "";
            if (categories != null)
            {
                foreach (var category in categories)
                {
                    var index = category.labels.FindIndex(x => x.spriteId == spriteId);
                    if (index == 0)
                    {
                        categoryName = category.name;
                        return true;
                    }
                    if (index > 0)
                        return false;
                }
            }
            return true;
        } 
#endif                
    }
}
