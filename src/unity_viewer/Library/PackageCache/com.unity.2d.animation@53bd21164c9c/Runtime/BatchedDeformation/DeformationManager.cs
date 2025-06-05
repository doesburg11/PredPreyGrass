using System.Collections.Generic;

namespace UnityEngine.U2D.Animation
{
    /// <summary>
    /// The available modes for batched Sprite Skin deformation.
    /// </summary>
    public enum DeformationMethods
    {
        /// <summary>
        /// The Sprite Skin is deformed, in batch, on the CPU.
        /// </summary>
        Cpu = 0,
        /// <summary>
        /// The Sprite Skin is deformed, in batch, on the GPU.
        /// </summary>
        Gpu = 1,
        /// <summary>
        /// Used as a default value when no deformation method is chosen.
        /// </summary>
        None = 2
    }

    internal class DeformationManager : ScriptableObject
    {
        static DeformationManager s_Instance;

        public static DeformationManager instance
        {
            get
            {
                if (s_Instance == null)
                {
                    var managers = Resources.FindObjectsOfTypeAll<DeformationManager>();
                    if (managers.Length > 0)
                        s_Instance = managers[0];
                    else
                        s_Instance = ScriptableObject.CreateInstance<DeformationManager>();
                    s_Instance.hideFlags = HideFlags.HideAndDontSave;
                    s_Instance.Init();
                }

                return s_Instance;
            }
        }

        BaseDeformationSystem[] m_DeformationSystems;

        [SerializeField]
        GameObject m_Helper;
        internal GameObject helperGameObject => m_Helper;

        bool canUseGpuDeformation { get; set; }
        bool m_WasUsingGpuDeformationLastFrame;

        void OnEnable()
        {
            s_Instance = this;
            canUseGpuDeformation = SpriteSkinUtility.CanUseGpuDeformation();
            m_WasUsingGpuDeformationLastFrame = SpriteSkinUtility.IsUsingGpuDeformation();

            Init();
        }

        void Init()
        {
            CreateBatchSystems();
            CreateHelper();
        }

        void CreateBatchSystems()
        {
            if (m_DeformationSystems != null)
                return;

            var noOfSystems = canUseGpuDeformation ? 2 : 1;
            m_DeformationSystems = new BaseDeformationSystem[noOfSystems];
            m_DeformationSystems[(int)DeformationMethods.Cpu] = new CpuDeformationSystem();

            if (canUseGpuDeformation)
                m_DeformationSystems[(int)DeformationMethods.Gpu] = new GpuDeformationSystem();

            for (var i = 0; i < m_DeformationSystems.Length; ++i)
                m_DeformationSystems[i].Initialize(m_DeformationSystems[i].GetHashCode());
        }

        void CreateHelper()
        {
            if (m_Helper != null)
                return;

            m_Helper = new GameObject("DeformationManagerUpdater");
            m_Helper.hideFlags = HideFlags.HideAndDontSave;
            var helperComponent = m_Helper.AddComponent<DeformationManagerUpdater>();
            helperComponent.onDestroyingComponent += OnHelperDestroyed;

#if !UNITY_EDITOR
            GameObject.DontDestroyOnLoad(m_Helper);
#endif
        }

        void OnHelperDestroyed(GameObject helperGo)
        {
            if (m_Helper != helperGo)
                return;

            m_Helper = null;
            CreateHelper();
        }

        void OnDisable()
        {
            if (m_Helper != null)
            {
                m_Helper.GetComponent<DeformationManagerUpdater>().onDestroyingComponent -= OnHelperDestroyed;
                GameObject.DestroyImmediate(m_Helper);
            }

            for (var i = 0; i < m_DeformationSystems.Length; ++i)
                m_DeformationSystems[i].Cleanup();
        }

        internal void Update()
        {
            if (HasToggledGpuDeformation())
                MoveSpriteSkinsToActiveSystem();

            for (var i = 0; i < m_DeformationSystems.Length; ++i)
                m_DeformationSystems[i].Update();
        }

        bool HasToggledGpuDeformation()
        {
            var isUsingGpuDeformation = SpriteSkinUtility.IsUsingGpuDeformation();
            if (isUsingGpuDeformation != m_WasUsingGpuDeformationLastFrame)
            {
                m_WasUsingGpuDeformationLastFrame = isUsingGpuDeformation;
                return true;
            }

            return false;
        }

        void MoveSpriteSkinsToActiveSystem()
        {
            var prevSystem = SpriteSkinUtility.IsUsingGpuDeformation() ? m_DeformationSystems[(int)DeformationMethods.Cpu] : m_DeformationSystems[(int)DeformationMethods.Gpu];

            var skins = prevSystem.GetSpriteSkins();
            foreach (var spriteSkin in skins)
                prevSystem.RemoveSpriteSkin(spriteSkin);

            foreach (var spriteSkin in skins)
                AddSpriteSkin(spriteSkin);
        }

        internal void AddSpriteSkin(SpriteSkin spriteSkin)
        {
            if (spriteSkin == null)
                return;

            var deformationMethod = SpriteSkinUtility.IsUsingGpuDeformation() ? DeformationMethods.Gpu : DeformationMethods.Cpu;
            if (deformationMethod == DeformationMethods.Gpu)
            {
                if (!canUseGpuDeformation)
                {
                    deformationMethod = DeformationMethods.Cpu;
                    Debug.LogWarning($"{spriteSkin.name} is trying to use GPU deformation, but the platform does not support it. Switching the renderer over to CPU deformation.", spriteSkin);
                }
                else if (!SpriteSkinUtility.CanSpriteSkinUseGpuDeformation(spriteSkin))
                {
                    deformationMethod = DeformationMethods.Cpu;
                    Debug.LogWarning($"{spriteSkin.name} is using a shader without GPU deformation support. Switching the renderer over to CPU deformation.", spriteSkin);
                }
            }

            var deformationSystem = m_DeformationSystems[(int)deformationMethod];
            if (deformationSystem.AddSpriteSkin(spriteSkin))
                spriteSkin.SetDeformationSystem(deformationSystem);
        }

        internal void RemoveBoneTransforms(SpriteSkin spriteSkin)
        {
            for (var i = 0; i < m_DeformationSystems.Length; ++i)
                m_DeformationSystems[i].RemoveBoneTransforms(spriteSkin);
        }

        internal void AddSpriteSkinBoneTransform(SpriteSkin spriteSkin)
        {
            if (spriteSkin == null)
                return;
            var system = spriteSkin.deformationSystem;
            if (system == null)
                return;

            system.AddBoneTransforms(spriteSkin);
        }

#if UNITY_INCLUDE_TESTS
        internal SpriteSkin[] GetSpriteSkins()
        {
            var skinList = new List<SpriteSkin>();
            for (var i = 0; i < m_DeformationSystems.Length; ++i)
                skinList.AddRange(m_DeformationSystems[i].GetSpriteSkins());

            return skinList.ToArray();
        }

        internal TransformAccessJob GetWorldToLocalTransformAccessJob(DeformationMethods deformationMethod)
        {
            if (!IsValidDeformationMethod(deformationMethod, out var systemIndex))
                return null;
            return m_DeformationSystems[systemIndex].GetWorldToLocalTransformAccessJob();
        }

        internal TransformAccessJob GetLocalToWorldTransformAccessJob(DeformationMethods deformationMethod)
        {
            if (!IsValidDeformationMethod(deformationMethod, out var systemIndex))
                return null;
            return m_DeformationSystems[systemIndex].GetLocalToWorldTransformAccessJob();
        }

        bool IsValidDeformationMethod(DeformationMethods deformationMethod, out int methodIndex)
        {
            methodIndex = (int)deformationMethod;
            return methodIndex < m_DeformationSystems.Length;
        }
#endif
    }

#if UNITY_EDITOR

    [UnityEditor.InitializeOnLoad]
    internal class DeformationStartup
    {
        static DeformationStartup()
        {
            if (null == DeformationManager.instance.helperGameObject)
                throw new System.InvalidOperationException("SpriteSkinComposite not initialized properly.");
        }
    }

#endif

}