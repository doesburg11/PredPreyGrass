using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor.U2D.Common;
using UnityEditor.U2D.Layout;
using UnityEngine.U2D;

namespace UnityEditor.U2D.Animation
{
    internal interface ICopyToolStringStore
    {
        string stringStore { get; set; }
    }

    internal class SystemCopyBufferStringStore : ICopyToolStringStore
    {
        public string stringStore
        {
            get => EditorGUIUtility.systemCopyBuffer;
            set => EditorGUIUtility.systemCopyBuffer = value;
        }
    }

    internal class BoneStorage
    {
        public BoneCache[] bones;
        public Dictionary<string, string> boneMapping;

        public BoneStorage(BoneCache[] bones, Dictionary<string, string> boneMapping = null)
        {
            this.bones = bones;
            this.boneMapping = boneMapping ?? new Dictionary<string, string>();
        }
    }

    internal class CopyTool : MeshToolWrapper
    {
        ICopyToolStringStore m_CopyToolStringStore;
        CopyToolView m_CopyToolView;
        bool m_HasValidCopyData = false;
        int m_LastCopyDataHash;

        public float pixelsPerUnit { private get; set; }

        public bool hasValidCopiedData
        {
            get
            {
                var hashCode = m_CopyToolStringStore.stringStore.GetHashCode();
                if (hashCode != m_LastCopyDataHash)
                {
                    m_HasValidCopyData = IsValidCopyData(m_CopyToolStringStore.stringStore);
                    m_LastCopyDataHash = hashCode;
                }

                return m_HasValidCopyData;
            }
        }

        public ICopyToolStringStore copyToolStringStore
        {
            set => m_CopyToolStringStore = value;
        }

        internal override void OnCreate()
        {
            m_CopyToolView = new CopyToolView();
            m_CopyToolView.onPasteActivated += OnPasteActivated;
            m_CopyToolStringStore = new SystemCopyBufferStringStore();
            disableMeshEditor = true;
        }

        public override void Initialize(LayoutOverlay layout)
        {
            m_CopyToolView.Initialize(layout);
        }

        protected override void OnActivate()
        {
            base.OnActivate();
            m_CopyToolView.Show(skinningCache.bonesReadOnly);
        }

        protected override void OnDeactivate()
        {
            base.OnDeactivate();
            m_CopyToolView.Hide();
        }

        void CopyMeshFromSpriteCache(SpriteCache sprite, SkinningCopySpriteData skinningSpriteData)
        {
            if (meshTool == null)
                return;

            meshTool.SetupSprite(sprite);
            skinningSpriteData.vertices = meshTool.mesh.vertices;
            skinningSpriteData.vertexWeights = meshTool.mesh.vertexWeights;
            skinningSpriteData.indices = meshTool.mesh.indices;
            skinningSpriteData.edges = meshTool.mesh.edges;
            skinningSpriteData.boneWeightGuids = new List<string>(meshTool.mesh.bones.Length);
            skinningSpriteData.boneWeightNames = new List<string>(meshTool.mesh.bones.Length);
            foreach (var bone in meshTool.mesh.bones)
            {
                skinningSpriteData.boneWeightGuids.Add(bone.guid);
                skinningSpriteData.boneWeightNames.Add(bone.name);
            }
        }

        public void OnCopyActivated()
        {
            SkinningCopyData skinningCopyData = null;
            var selectedSprite = skinningCache.selectedSprite;
            if (selectedSprite == null)
            {
                var sprites = skinningCache.GetSprites();
                if (!skinningCache.character || sprites.Length > 1)
                    skinningCopyData = CopyAll();
                else if (sprites.Length == 1)
                    skinningCopyData = CopySingle(sprites[0]);
            }
            else
            {
                skinningCopyData = CopySingle(selectedSprite);
            }

            if (skinningCopyData != null)
                m_CopyToolStringStore.stringStore = SkinningCopyUtility.SerializeSkinningCopyDataToString(skinningCopyData);
            skinningCache.events.copy.Invoke();
        }

        SkinningCopyData CopyAll()
        {
            var skinningCopyData = new SkinningCopyData
            {
                pixelsPerUnit = pixelsPerUnit,
                isCharacterData = skinningCache.hasCharacter,
                characterBones = skinningCache.hasCharacter
                    ? skinningCache.character.skeleton.bones.ToSpriteBone(Matrix4x4.identity)
                    : null
            };

            var sprites = skinningCache.GetSprites();
            foreach (var sprite in sprites)
                skinningCopyData.copyData.Add(GetDataForSprite(sprite));

            if (meshTool != null)
                meshTool.SetupSprite(null);

            return skinningCopyData;
        }

        SkinningCopyData CopySingle(SpriteCache sprite)
        {
            var skinningCopyData = new SkinningCopyData
            {
                pixelsPerUnit = pixelsPerUnit,
                isCharacterData = false,
                characterBones = Array.Empty<SpriteBone>()
            };
            skinningCopyData.copyData.Add(GetDataForSprite(sprite));

            return skinningCopyData;
        }

        SkinningCopySpriteData GetDataForSprite(SpriteCache sprite)
        {
            var skinningSpriteData = new SkinningCopySpriteData();
            skinningSpriteData.spriteName = sprite.name;

            CopyMeshFromSpriteCache(sprite, skinningSpriteData);

            // Bones
            var rootBones = new List<BoneCache>();
            var boneCache = Array.Empty<BoneCache>();
            if (skinningCache.hasCharacter)
            {
                var characterPart = skinningCache.GetCharacterPart(sprite);
                if (characterPart != null && characterPart.bones != null)
                {
                    boneCache = characterPart.bones;
                    var bones = characterPart.bones.FindRoots();
                    foreach (var bone in bones)
                        rootBones.Add(bone);
                }
            }
            else
            {
                var skeleton = skinningCache.GetEffectiveSkeleton(sprite);
                if (skeleton != null && skeleton.boneCount > 0)
                {
                    boneCache = skeleton.bones;
                    var bones = boneCache.FindRoots();
                    foreach (var bone in bones)
                        rootBones.Add(bone);
                }
            }

            if (rootBones.Count > 0)
            {
                skinningSpriteData.spriteBones = new List<SpriteBoneCopyData>();
                foreach (var rootBone in rootBones)
                {
                    var rootBoneIndex = skinningSpriteData.spriteBones.Count;

                    GetSpriteBoneDataRecursively(skinningSpriteData.spriteBones, rootBone, new List<BoneCache>(boneCache));
                    if (skinningCache.hasCharacter)
                    {
                        // Offset the bones based on the currently selected Sprite in Character mode
                        var characterPart = sprite.GetCharacterPart();
                        if (characterPart != null)
                        {
                            var offset = characterPart.position;
                            var rootSpriteBone = skinningSpriteData.spriteBones[rootBoneIndex];
                            rootSpriteBone.spriteBone.position -= offset;
                            skinningSpriteData.spriteBones[rootBoneIndex] = rootSpriteBone;
                        }
                    }
                }
            }

            return skinningSpriteData;
        }

        static void GetSpriteBoneDataRecursively(IList<SpriteBoneCopyData> bones, BoneCache rootBone, List<BoneCache> boneCache)
        {
            AppendSpriteBoneDataRecursively(bones, rootBone, -1, boneCache);
        }

        static void AppendSpriteBoneDataRecursively(IList<SpriteBoneCopyData> bones, BoneCache currentBone, int parentIndex, List<BoneCache> boneCache)
        {
            var currentParentIndex = bones.Count;

            var boneCopyData = new SpriteBoneCopyData()
            {
                spriteBone = new SpriteBone()
                {
                    name = currentBone.name,
                    guid = currentBone.guid,
                    color = currentBone.bindPoseColor,
                    parentId = parentIndex
                },
                order = boneCache.FindIndex(x => x == currentBone)
            };
            if (boneCopyData.order < 0)
            {
                boneCopyData.order = boneCache.Count;
                boneCache.Add(currentBone);
            }

            if (parentIndex == -1 && currentBone.parentBone != null)
            {
                boneCopyData.spriteBone.position = currentBone.position;
                boneCopyData.spriteBone.rotation = currentBone.rotation;
            }
            else
            {
                boneCopyData.spriteBone.position = currentBone.localPosition;
                boneCopyData.spriteBone.rotation = currentBone.localRotation;
            }

            boneCopyData.spriteBone.position = new Vector3(boneCopyData.spriteBone.position.x, boneCopyData.spriteBone.position.y, currentBone.depth);

            boneCopyData.spriteBone.length = currentBone.localLength;
            bones.Add(boneCopyData);
            foreach (var child in currentBone)
            {
                var childBone = child as BoneCache;
                if (childBone != null)
                    AppendSpriteBoneDataRecursively(bones, childBone, currentParentIndex, boneCache);
            }
        }

        public void OnPasteActivated(bool shouldPasteBones, bool shouldPasteMesh, bool shouldFlipX, bool shouldFlipY)
        {
            var copyBuffer = m_CopyToolStringStore.stringStore;
            if (!IsValidCopyData(copyBuffer))
                return;

            var skinningCopyData = SkinningCopyUtility.DeserializeStringToSkinningCopyData(copyBuffer);
            if (skinningCopyData == null || skinningCopyData.copyData.Count == 0)
                return;

            var doesCopyContainMultipleSprites = skinningCopyData.copyData.Count > 1;
            var sprites = skinningCache.GetSprites();

            if (doesCopyContainMultipleSprites && skinningCopyData.copyData.Count != sprites.Length && shouldPasteMesh)
            {
                Debug.Log(string.Format(TextContent.copyIncorrectNumberOfSprites, skinningCopyData.copyData.Count, sprites.Length));
                return;
            }

            var selectedSprite = skinningCache.selectedSprite;
            using (skinningCache.UndoScope(TextContent.pasteData))
            {
                var scale = skinningCopyData.pixelsPerUnit > 0f ? pixelsPerUnit / skinningCopyData.pixelsPerUnit : 1f;
                var pastedBonesToSelect = new HashSet<BoneCache>();

                var characterBones = Array.Empty<BoneCache>();
                var replaceCharacterSkeleton = shouldPasteBones && skinningCache.hasCharacter && skinningCopyData.isCharacterData;
                if (replaceCharacterSkeleton)
                {
                    var spriteBones = skinningCopyData.characterBones;
                    characterBones = PasteBonesInCharacter(skinningCache, spriteBones, shouldFlipX, shouldFlipY, scale);
                    foreach (var newBone in characterBones)
                        pastedBonesToSelect.Add(newBone);
                }

                var pastedToSprites = new List<SpriteCache>();
                foreach (var copySpriteData in skinningCopyData.copyData)
                {
                    SpriteCache sprite = null;
                    if (selectedSprite != null && !doesCopyContainMultipleSprites)
                        sprite = selectedSprite;
                    if (sprite == null && !string.IsNullOrEmpty(copySpriteData.spriteName))
                        sprite = FindSpriteWithName(sprites.Except(pastedToSprites).ToList(), copySpriteData.spriteName) ?? FindSpriteWithName(sprites, copySpriteData.spriteName);

                    if (sprite == null)
                        continue;

                    pastedToSprites.Add(sprite);

                    var boneMapping = new Dictionary<string, string>();
                    if (shouldPasteBones && !replaceCharacterSkeleton)
                    {
                        var bonesToPaste = GetBonesInCorrectOrder(copySpriteData.spriteBones);
                        var boneStorage = PasteBonesInSprite(skinningCache, sprite, bonesToPaste, characterBones, shouldFlipX, shouldFlipY, scale);
                        if (boneStorage != null)
                        {
                            boneMapping = boneStorage.boneMapping;
                            if (skinningCache.hasCharacter || sprite == selectedSprite)
                            {
                                foreach (var newBone in boneStorage.bones)
                                    pastedBonesToSelect.Add(newBone);
                            }
                        }
                    }

                    if (shouldPasteMesh)
                        PasteMeshInSprite(meshTool, sprite, copySpriteData, shouldFlipX, shouldFlipY, scale, boneMapping);
                }

                var refreshSelection = skinningCache.hasCharacter || skinningCache.selectedSprite != null;
                if (refreshSelection)
                {
                    var newBoneSelection = new BoneCache[pastedBonesToSelect.Count];
                    pastedBonesToSelect.CopyTo(newBoneSelection);

                    meshTool.SetupSprite(selectedSprite); // This is to refresh the selected Sprite in meshTool.
                    skinningCache.skeletonSelection.elements = newBoneSelection;
                    skinningCache.events.boneSelectionChanged.Invoke();
                }
            }

            skinningCache.events.paste.Invoke(shouldPasteBones, shouldPasteMesh, shouldFlipX, shouldFlipY);
        }

        static bool IsValidCopyData(string copyBuffer)
        {
            return SkinningCopyUtility.CanDeserializeStringToSkinningCopyData(copyBuffer);
        }

        static Vector3 GetFlippedBonePosition(BoneCache bone, Vector2 startPosition, Rect spriteRect, bool flipX, bool flipY)
        {
            Vector3 position = startPosition;
            if (flipX)
                position.x += spriteRect.width - bone.position.x;
            else
                position.x += bone.position.x;

            if (flipY)
                position.y += spriteRect.height - bone.position.y;
            else
                position.y += bone.position.y;

            position.z = bone.position.z;
            return position;
        }

        static Quaternion GetFlippedBoneRotation(BoneCache bone, bool flipX, bool flipY)
        {
            var euler = bone.rotation.eulerAngles;
            if (flipX)
            {
                if (euler.z <= 180)
                    euler.z = 180 - euler.z;
                else
                    euler.z = 540 - euler.z;
            }

            if (flipY)
                euler.z = 360 - euler.z;
            return Quaternion.Euler(euler);
        }

        static void SetBonePositionAndRotation(BoneCache[] boneCache, TransformCache bone, Vector3[] position, Quaternion[] rotation)
        {
            var index = Array.FindIndex(boneCache, x => x == bone);
            if (index >= 0)
            {
                bone.position = position[index];
                bone.rotation = rotation[index];
            }

            foreach (var child in bone.children)
            {
                SetBonePositionAndRotation(boneCache, child, position, rotation);
            }
        }

        static BoneCache[] PasteBonesInCharacter(SkinningCache skinningCache, SpriteBone[] spriteBones, bool shouldFlipX, bool shouldFlipY, float scale)
        {
            if (!skinningCache.hasCharacter)
                return null;

            var boneCache = skinningCache.CreateBoneCacheFromSpriteBones(spriteBones, scale);
            if (shouldFlipX || shouldFlipY)
            {
                var characterRect = new Rect(Vector2.zero, skinningCache.character.dimension);
                var newPositions = new Vector3[boneCache.Length];
                var newRotations = new Quaternion[boneCache.Length];
                for (var i = 0; i < boneCache.Length; ++i)
                {
                    newPositions[i] = GetFlippedBonePosition(boneCache[i], Vector2.zero, characterRect, shouldFlipX, shouldFlipY);
                    newRotations[i] = GetFlippedBoneRotation(boneCache[i], shouldFlipX, shouldFlipY);
                }

                for (var i = 0; i < boneCache.Length; ++i)
                {
                    boneCache[i].position = newPositions[i];
                    boneCache[i].rotation = newRotations[i];
                }
            }

            var skeleton = skinningCache.character.skeleton;
            skeleton.SetBones(boneCache);
            skinningCache.events.skeletonTopologyChanged.Invoke(skeleton);

            return boneCache;
        }

        static BoneStorage PasteBonesInSprite(SkinningCache skinningCache, SpriteCache sprite, SpriteBone[] newBones, BoneCache[] characterBones, bool shouldFlipX, bool shouldFlipY, float scale)
        {
            if (sprite == null || skinningCache.mode == SkinningMode.SpriteSheet && skinningCache.hasCharacter)
                return null;

            var spriteRect = sprite.textureRect;
            var skeleton = skinningCache.GetEffectiveSkeleton(sprite);

            var newBonesCache = skinningCache.CreateBoneCacheFromSpriteBones(newBones, scale);
            if (newBonesCache.Length == 0)
                return null;

            Vector2 rectPosition;
            if (skinningCache.mode == SkinningMode.Character)
            {
                var characterPart = sprite.GetCharacterPart();
                if (characterPart == null)
                    return null;
                rectPosition = characterPart.position;
            }
            else
                rectPosition = spriteRect.position;

            var newPositions = new Vector3[newBonesCache.Length];
            var newRotations = new Quaternion[newBonesCache.Length];
            for (var i = 0; i < newBonesCache.Length; ++i)
            {
                newPositions[i] = GetFlippedBonePosition(newBonesCache[i], rectPosition, spriteRect, shouldFlipX, shouldFlipY);
                newRotations[i] = GetFlippedBoneRotation(newBonesCache[i], shouldFlipX, shouldFlipY);
            }

            foreach (var bone in newBonesCache)
            {
                if (bone.parent == null)
                {
                    SetBonePositionAndRotation(newBonesCache, bone, newPositions, newRotations);
                    if (skinningCache.mode == SkinningMode.Character)
                        bone.SetParent(skeleton);
                }
            }

            var boneNameMapping = new Dictionary<string, string>();
            if (skinningCache.mode == SkinningMode.SpriteSheet)
            {
                skeleton.SetBones(newBonesCache);
                skeleton.SetDefaultPose();
            }
            else
            {
                boneNameMapping = AddBonesToSkeletonWithUniqueNames(characterBones, newBonesCache, skeleton);
                skeleton.SetDefaultPose();
            }

            skinningCache.events.skeletonTopologyChanged.Invoke(skeleton);
            return new BoneStorage(newBonesCache, boneNameMapping);
        }

        static Dictionary<string, string> AddBonesToSkeletonWithUniqueNames(IList<BoneCache> characterBones, IList<BoneCache> newBones, SkeletonCache skeleton)
        {
            var nameMapping = new Dictionary<string, string>();

            var existingBoneGuids = new HashSet<string>();
            var existingBoneNames = new HashSet<string>(skeleton.boneCount);
            for (var i = 0; i < characterBones.Count; i++)
            {
                if (!string.IsNullOrEmpty(characterBones[i].guid))
                    existingBoneGuids.Add(characterBones[i].guid);
            }

            for (var i = 0; i < skeleton.boneCount; i++)
                existingBoneNames.Add(skeleton.bones[i].name);

            foreach (var newBone in newBones)
            {
                var guid = newBone.guid;
                if (string.IsNullOrEmpty(guid) || existingBoneGuids.Contains(guid))
                    continue;

                var boneName = newBone.name;
                if (existingBoneNames.Contains(boneName))
                    newBone.name = SkeletonController.AutoNameBoneCopy(boneName, skeleton.bones);

                existingBoneGuids.Add(newBone.guid);
                existingBoneNames.Add(newBone.name);
                nameMapping[boneName] = newBone.name;
                skeleton.AddBone(newBone);
            }

            return nameMapping;
        }

        static void PasteMeshInSprite(MeshTool meshTool, SpriteCache sprite, SkinningCopySpriteData copySpriteData, bool shouldFlipX, bool shouldFlipY, float scale, Dictionary<string, string> boneMapping)
        {
            if (meshTool == null || sprite == null)
                return;

            var vertices = copySpriteData.vertices ?? Array.Empty<Vector2>();
            var vertexWeights = copySpriteData.vertexWeights ?? Array.Empty<EditableBoneWeight>();

            meshTool.SetupSprite(sprite);
            meshTool.mesh.SetVertices(vertices, vertexWeights);
            if (!Mathf.Approximately(scale, 1f) || shouldFlipX || shouldFlipY)
            {
                var spriteRect = sprite.textureRect;
                for (var i = 0; i < meshTool.mesh.vertexCount; ++i)
                {
                    var position = meshTool.mesh.vertices[i];
                    if (!Mathf.Approximately(scale, 1f))
                        position *= scale;
                    if (shouldFlipX)
                        position.x = spriteRect.width - meshTool.mesh.vertices[i].x;
                    if (shouldFlipY)
                        position.y = spriteRect.height - meshTool.mesh.vertices[i].y;
                    meshTool.mesh.vertices[i] = position;
                }
            }

            meshTool.mesh.SetIndices(copySpriteData.indices);
            meshTool.mesh.SetEdges(copySpriteData.edges);

            var skinningCache = meshTool.skinningCache;
            var skeleton = skinningCache.GetEffectiveSkeleton(sprite);
            var hasGuids = copySpriteData.boneWeightGuids.Count > 0;
            for (var i = 0; i < copySpriteData.boneWeightGuids.Count; i++)
            {
                if (string.IsNullOrEmpty(copySpriteData.boneWeightGuids[i]))
                {
                    hasGuids = false;
                    break;
                }
            }

            var skeletonBones = skeleton.bones;
            var influenceBones = hasGuids ? GetBonesFromGuids(copySpriteData, skeletonBones, boneMapping) : GetBonesFromNames(copySpriteData, skeletonBones, boneMapping);

            // Update associated bones for mesh
            meshTool.mesh.SetCompatibleBoneSet(influenceBones);
            meshTool.mesh.bones = influenceBones; // Fixes weights for bones that do not exist

            // Update associated bones for character
            if (skinningCache.hasCharacter)
            {
                var characterPart = sprite.GetCharacterPart();
                if (characterPart != null)
                {
                    characterPart.bones = influenceBones;
                    skinningCache.events.characterPartChanged.Invoke(characterPart);
                }
            }

            meshTool.UpdateMesh();
        }

        static BoneCache[] GetBonesFromGuids(SkinningCopySpriteData copySpriteData, IList<BoneCache> skeletonBones, Dictionary<string, string> boneMapping)
        {
            var spriteBones = new List<BoneCache>();
            for (var i = 0; i < copySpriteData.boneWeightGuids.Count; i++)
            {
                var bone = FindBoneWithGuid(skeletonBones, copySpriteData.boneWeightGuids[i]);
                if (bone == null)
                    continue;

                if (boneMapping != null && boneMapping.ContainsKey(bone.name))
                {
                    bone = FindBoneWithName(skeletonBones, boneMapping[bone.name]);
                    if (bone == null)
                        continue;
                }

                spriteBones.Add(bone);
            }

            return spriteBones.ToArray();
        }

        static BoneCache[] GetBonesFromNames(SkinningCopySpriteData copySpriteData, IList<BoneCache> skeletonBones, Dictionary<string, string> boneMapping)
        {
            var spriteBones = new List<BoneCache>();
            for (var i = 0; i < copySpriteData.boneWeightNames.Count; ++i)
            {
                var boneName = copySpriteData.boneWeightNames[i];
                if (boneMapping != null && boneMapping.ContainsKey(boneName))
                    boneName = boneMapping[boneName];

                var bone = FindBoneWithName(skeletonBones, boneName);
                if (bone == null)
                    continue;

                spriteBones.Add(bone);
            }

            return spriteBones.ToArray();
        }

        static SpriteBone[] GetBonesInCorrectOrder(IList<SpriteBoneCopyData> spriteBones)
        {
            var orderedBones = new SpriteBone[spriteBones.Count];
            for (var i = 0; i < spriteBones.Count; ++i)
            {
                var order = spriteBones[i].order;
                if (order >= 0)
                {
                    orderedBones[order] = spriteBones[i].spriteBone;
                    var parentId = orderedBones[order].parentId;
                    if (parentId >= 0)
                        orderedBones[order].parentId = spriteBones[parentId].order;
                }
                else
                {
                    orderedBones[i] = spriteBones[i].spriteBone;
                }
            }

            return orderedBones;
        }

        static SpriteCache FindSpriteWithName(IList<SpriteCache> sprites, string spriteName)
        {
            for (var i = 0; i < sprites.Count; i++)
            {
                var sprite = sprites[i];
                if (sprite.name == spriteName)
                    return sprite;
            }

            return null;
        }

        static BoneCache FindBoneWithName(IList<BoneCache> bones, string boneName)
        {
            for (var i = 0; i < bones.Count; i++)
            {
                var bone = bones[i];
                if (bone.name == boneName)
                    return bone;
            }

            return null;
        }

        static BoneCache FindBoneWithGuid(IList<BoneCache> bones, string guid)
        {
            for (var i = 0; i < bones.Count; i++)
            {
                var bone = bones[i];
                if (bone.guid == guid)
                    return bone;
            }

            return null;
        }
    }

    internal class CopyToolView
    {
        PastePanel m_PastePanel;

        public event Action<bool, bool, bool, bool> onPasteActivated = (bone, mesh, flipX, flipY) => { };

        public void Show(bool readonlyBone)
        {
            m_PastePanel.SetHiddenFromLayout(false);
            m_PastePanel.BonePasteEnable(!readonlyBone);
        }

        public void Hide()
        {
            m_PastePanel.SetHiddenFromLayout(true);
        }

        public void Initialize(LayoutOverlay layoutOverlay)
        {
            m_PastePanel = PastePanel.GenerateFromUXML();
            BindElements();
            layoutOverlay.rightOverlay.Add(m_PastePanel);
            m_PastePanel.SetHiddenFromLayout(true);
        }

        void BindElements()
        {
            m_PastePanel.onPasteActivated += OnPasteActivated;
        }

        void OnPasteActivated(bool bone, bool mesh, bool flipX, bool flipY)
        {
            onPasteActivated(bone, mesh, flipX, flipY);
        }
    }
}
