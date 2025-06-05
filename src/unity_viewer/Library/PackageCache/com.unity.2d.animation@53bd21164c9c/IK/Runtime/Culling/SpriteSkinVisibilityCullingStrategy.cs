using System;
using System.Collections.Generic;
using UnityEngine.U2D.Animation;

namespace UnityEngine.U2D.IK
{
    internal class SpriteSkinVisibilityCullingStrategy : BaseCullingStrategy
    {
        /// <summary>
        /// SpriteSkin registry used to keep visibility state of a SpriteSkin and bone transforms.
        /// </summary>
        class SpriteSkinRegistry
        {
            public int[] boneIds;
            public bool isVisible;

            public SpriteSkinRegistry(int[] boneIds, bool isSkinVisible)
            {
                this.boneIds = boneIds;
                isVisible = isSkinVisible;
            }
        }

        /// <summary>
        /// Maps SpriteSkins to SpriteSkinRegistry.
        /// </summary>
        Dictionary<SpriteSkin, SpriteSkinRegistry> m_SpriteSkinRegistries;

        /// <summary>
        /// Counts (value) how many visible Sprite Skins use a given bone (key).
        /// </summary>
        Dictionary<int, int> m_BoneVisibilityCount;

        public override bool AreBonesVisible(IList<int> boneTransformIds)
        {
            for (var i = 0; i < boneTransformIds.Count; i++)
            {
                var boneId = boneTransformIds[i];
                if (m_BoneVisibilityCount.ContainsKey(boneId))
                    return m_BoneVisibilityCount[boneId] > 0;
            }

            return false;
        }

        protected override void OnInitialize()
        {
            m_SpriteSkinRegistries = new Dictionary<SpriteSkin, SpriteSkinRegistry>();
            m_BoneVisibilityCount = new Dictionary<int, int>();

            var spriteSkins = SpriteSkinContainer.instance.spriteSkins;
            for (var i = 0; i < spriteSkins.Count; i++)
                UpdateSpriteSkinVisibility(spriteSkins[i]);

            AddListeners();
        }

        protected override void OnDisable()
        {
            m_SpriteSkinRegistries.Clear();
            m_BoneVisibilityCount.Clear();

            RemoveListeners();
        }

        void AddListeners()
        {
            SpriteSkinContainer.onAddedSpriteSkin += UpdateSpriteSkinVisibility;
            SpriteSkinContainer.onRemovedSpriteSkin += UnregisterSpriteSkin;
            SpriteSkinContainer.onBoneTransformChanged += OnBoneTransformChanged;
        }

        void RemoveListeners()
        {
            SpriteSkinContainer.onAddedSpriteSkin -= UpdateSpriteSkinVisibility;
            SpriteSkinContainer.onRemovedSpriteSkin -= UnregisterSpriteSkin;
            SpriteSkinContainer.onBoneTransformChanged -= OnBoneTransformChanged;
        }

        protected override void OnUpdate()
        {
            foreach (var (spriteSkin, registry) in m_SpriteSkinRegistries)
            {
                var isVisible = spriteSkin.spriteRenderer.isVisible;
                if (registry.isVisible != isVisible)
                {
                    registry.isVisible = isVisible;
                    RecalculateVisibility(registry);
                }
            }
        }

        void OnBoneTransformChanged(SpriteSkin spriteSkin)
        {
            UnregisterSpriteSkinBonesMapping(spriteSkin);
            RegisterSpriteSkinBonesMapping(spriteSkin);

            UpdateSpriteSkinVisibility(spriteSkin);
        }

        bool IsSpriteSkinRegistered(SpriteSkin spriteSkin) => m_SpriteSkinRegistries.ContainsKey(spriteSkin);

        void UnregisterSpriteSkin(SpriteSkin spriteSkin)
        {
            UnregisterSpriteSkinBonesMapping(spriteSkin);
        }

        void UpdateSpriteSkinVisibility(SpriteSkin spriteSkin)
        {
            var visible = spriteSkin.spriteRenderer.isVisible;
            var registry = RegisterSpriteSkinBonesMapping(spriteSkin);
            if (registry.isVisible == visible)
                return;

            registry.isVisible = visible;

            RecalculateVisibility(registry);
        }

        SpriteSkinRegistry RegisterSpriteSkinBonesMapping(SpriteSkin spriteSkin)
        {
            if (IsSpriteSkinRegistered(spriteSkin))
                return m_SpriteSkinRegistries[spriteSkin];

            var bones = spriteSkin.boneTransforms ?? Array.Empty<Transform>();
            var records = new int[bones.Length];
            var newRegistry = new SpriteSkinRegistry(records, false);
            for (var i = 0; i < bones.Length; i++)
            {
                var bone = bones[i];
                if (bone == null)
                    continue;
                var id = bone.GetInstanceID();
                records[i] = id;
            }

            m_SpriteSkinRegistries[spriteSkin] = newRegistry;
            return newRegistry;
        }

        void UnregisterSpriteSkinBonesMapping(SpriteSkin spriteSkin)
        {
            if (!IsSpriteSkinRegistered(spriteSkin))
                return;

            var registry = m_SpriteSkinRegistries[spriteSkin];
            registry.isVisible = false;

            m_SpriteSkinRegistries.Remove(spriteSkin);

            RecalculateVisibility(registry);
        }

        void RecalculateVisibility(SpriteSkinRegistry registry)
        {
            var bones = registry.boneIds;

            var visible = registry.isVisible;
            var countOperation = visible ? 1 : -1;

            for (var i = 0; i < bones.Length; i++)
            {
                var bone = bones[i];
                if (m_BoneVisibilityCount.ContainsKey(bone))
                {
                    var count = m_BoneVisibilityCount[bone] + countOperation;
                    if (count <= 0)
                        m_BoneVisibilityCount.Remove(bone);
                    else
                        m_BoneVisibilityCount[bone] = count;
                }
                else if (visible)
                    m_BoneVisibilityCount[bone] = 1;
            }
        }
    }
}