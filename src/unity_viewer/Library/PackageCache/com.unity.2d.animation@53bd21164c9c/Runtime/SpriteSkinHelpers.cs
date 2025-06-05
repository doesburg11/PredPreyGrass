using System;
using System.Collections.Generic;

namespace UnityEngine.U2D.Animation
{
    internal static class SpriteSkinHelpers
    {
        public static void CacheChildren(Transform current, Dictionary<int, List<SpriteSkin.TransformData>> cache)
        {
            var nameHash = current.name.GetHashCode();
            var entry = new SpriteSkin.TransformData()
            {
                fullName = string.Empty,
                transform = current
            };
            if (cache.TryGetValue(nameHash, out var value))
                value.Add(entry);
            else
                cache.Add(nameHash, new List<SpriteSkin.TransformData>(1) { entry });

            for (var i = 0; i < current.childCount; ++i)
                CacheChildren(current.GetChild(i), cache);
        }

        public static string GenerateTransformPath(Transform rootBone, Transform child)
        {
            var path = child.name;
            if (child == rootBone)
                return path;
            var parent = child.parent;
            do
            {
                path = parent.name + "/" + path;
                parent = parent.parent;
            } while (parent != rootBone && parent != null);

            return path;
        }

        public static bool GetSpriteBonesTransforms(SpriteSkin spriteSkin, out Transform[] outTransform)
        {
            var rootBone = spriteSkin.rootBone;
            var spriteBones = spriteSkin.sprite.GetBones();

            if (rootBone == null)
                throw new ArgumentException("rootBone parameter cannot be null");
            if (spriteBones == null)
                throw new ArgumentException("spriteBones parameter cannot be null");

            outTransform = new Transform[spriteBones.Length];

            var boneObjects = rootBone.GetComponentsInChildren<Bone>();
            if (boneObjects != null && boneObjects.Length >= spriteBones.Length)
            {
                using (SpriteSkin.Profiling.getSpriteBonesTransformFromGuid.Auto())
                {
                    var i = 0;
                    for (; i < spriteBones.Length; ++i)
                    {
                        var boneHash = spriteBones[i].guid;
                        var boneTransform = Array.Find(boneObjects, x => (x.guid == boneHash));
                        if (boneTransform == null)
                            break;

                        outTransform[i] = boneTransform.transform;
                    }

                    if (i >= spriteBones.Length)
                        return true;
                }
            }

            var hierarchyCache = spriteSkin.hierarchyCache;
            if (hierarchyCache.Count == 0)
                spriteSkin.CacheHierarchy();

            // If unable to successfully map via guid, fall back to path
            return GetSpriteBonesTransformFromPath(spriteBones, hierarchyCache, outTransform);
        }

        static bool GetSpriteBonesTransformFromPath(SpriteBone[] spriteBones, Dictionary<int, List<SpriteSkin.TransformData>> hierarchyCache, Transform[] outNewBoneTransform)
        {
            using (SpriteSkin.Profiling.getSpriteBonesTransformFromPath.Auto())
            {
                string[] bonePath = null;
                var foundBones = true;
                for (var i = 0; i < spriteBones.Length; ++i)
                {
                    var nameHash = spriteBones[i].name.GetHashCode();
                    if (!hierarchyCache.TryGetValue(nameHash, out var children))
                    {
                        outNewBoneTransform[i] = null;
                        foundBones = false;
                        continue;
                    }

                    if (children.Count == 1)
                        outNewBoneTransform[i] = children[0].transform;
                    else
                    {
                        if (bonePath == null)
                            bonePath = new string[spriteBones.Length];
                        if (bonePath[i] == null)
                            CalculateBoneTransformsPath(i, spriteBones, bonePath);

                        var m = 0;
                        for (; m < children.Count; ++m)
                        {
                            if (children[m].fullName.Contains(bonePath[i]))
                            {
                                outNewBoneTransform[i] = children[m].transform;
                                break;
                            }
                        }

                        if (m >= children.Count)
                        {
                            outNewBoneTransform[i] = null;
                            foundBones = false;
                        }
                    }
                }

                return foundBones;
            }
        }

        static void CalculateBoneTransformsPath(int index, SpriteBone[] spriteBones, string[] paths)
        {
            var spriteBone = spriteBones[index];
            var parentId = spriteBone.parentId;
            var bonePath = spriteBone.name;
            if (parentId != -1)
            {
                if (paths[parentId] == null)
                    CalculateBoneTransformsPath(spriteBone.parentId, spriteBones, paths);
                paths[index] = $"{paths[parentId]}/{bonePath}";
            }
            else
                paths[index] = bonePath;
        }
    }
}
