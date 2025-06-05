using System;
using System.Linq;
using UnityEditor.Callbacks;
using UnityEditor.Scripting.ScriptCompilation;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEditor.TestTools.TestRunner.GUI;
using UnityEngine;

namespace UnityEditor.TestTools.TestRunner
{
    /// <summary>
    /// The TestRunnerWindow class is repsonsible for drawing the Test Runner window.
    /// </summary>
    [Serializable]
    public class TestRunnerWindow : EditorWindow, IHasCustomMenu
    {
        private const string WindowTitle = "Test Runner";

        internal static class Styles
        {
            public static GUIStyle info;
            public static GUIStyle testList;

            static Styles()
            {
                info = new GUIStyle("CN Message");
                info.wordWrap = true;
                info.stretchHeight = true;
                info.margin.right = 15;

                testList = new GUIStyle("CN Box");
                testList.margin.top = 0;
                testList.padding.left = 3;
            }
        }

        private readonly GUIContent m_GUIHorizontalSplit = EditorGUIUtility.TrTextContent("Horizontal layout");
        private readonly GUIContent m_GUIVerticalSplit = EditorGUIUtility.TrTextContent("Vertical layout");
        private readonly GUIContent m_GUIDisablePlaymodeTestsRunner = EditorGUIUtility.TrTextContent("Disable playmode tests for all assemblies");
        private readonly GUIContent m_GUIRunPlayModeTestAsEditModeTests = EditorGUIUtility.TrTextContent("Run playmode tests as editmode tests");

        internal static TestRunnerWindow s_Instance;
        private bool m_IsBuilding;
        [NonSerialized]
        private bool m_Enabled;
        //internal TestFilterSettings filterSettings;

        [SerializeField]
        private SplitterState m_Spl = new SplitterState(new float[] { 75, 25 }, new[] { 32, 32 }, null);

        private TestRunnerWindowSettings m_Settings;

        private enum TestRunnerMenuLabels
        {
            EditMode = 0,
            PlayMode,
            Player
        }
        [SerializeField]
        private TestRunnerMenuLabels m_TestTypeToolbarIndex = TestRunnerMenuLabels.EditMode;
        internal TestListGUI m_SelectedTestTypes;
        [SerializeField]
        private TestListGUI[] m_TestListGUIs;

        private ITestRunnerApi m_testRunnerApi;

        private WindowResultUpdater m_WindowResultUpdater;

        /// <summary>
        /// Launches the Test Runner window.
        /// </summary>
        [MenuItem("Window/General/Test Runner", false, 201, false)]
        public static void ShowWindow()
        {
            s_Instance = GetWindow<TestRunnerWindow>(WindowTitle);
            s_Instance.Show();
        }

        static TestRunnerWindow()
        {
            InitBackgroundRunners();
        }

        private static void InitBackgroundRunners()
        {
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        [DidReloadScripts]
        private static void CompilationCallback()
        {
            UpdateWindow();
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (s_Instance && state == PlayModeStateChange.EnteredEditMode)
            {
                var testListGUI = s_Instance.m_SelectedTestTypes;
                if (testListGUI.HasTreeData())
                {
                    //repaint message details after exit playmode
                    testListGUI.TestSelectionCallback(testListGUI.m_TestListState.selectedIDs.ToArray());
                    s_Instance.Repaint();
                }
            }
        }

        internal void OnDestroy()
        {
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
        }

        private void OnEnable()
        {
            s_Instance = this;
            titleContent = new GUIContent(WindowTitle, "Test framework for running Edit mode and Play mode tests in Unity. Part of the com.unity.test-framework package.");
            SelectTestListGUI(m_TestTypeToolbarIndex);

            m_testRunnerApi = CreateInstance<TestRunnerApi>();
            m_WindowResultUpdater = new WindowResultUpdater();
            m_testRunnerApi.RegisterCallbacks(m_WindowResultUpdater);
        }

        private void Enable()
        {
            m_Settings = new TestRunnerWindowSettings("UnityEditor.PlaymodeTestsRunnerWindow");

            if (m_SelectedTestTypes == null)
            {
                SelectTestListGUI(m_TestTypeToolbarIndex);
            }

            StartRetrieveTestList();
            m_SelectedTestTypes.Reload();
            m_Enabled = true;
        }

        private void SelectTestListGUI(TestRunnerMenuLabels testTypeToolbarIndex)
        {
            if (m_TestListGUIs == null)
            {
                m_TestListGUIs = new TestListGUI[]
                {
                    new TestListGUI()
                    {
                        m_TestMode = TestMode.EditMode,
                    },
                    new TestListGUI()
                    {
                        m_TestMode = TestMode.PlayMode,
                    },
                    new TestListGUI()
                    {
                        m_TestMode = TestMode.PlayMode,
                        m_RunOnPlatform = true
                    }
                };
            }

            m_TestListGUIs[0].m_TestMode = TestMode.EditMode;
            m_TestListGUIs[0].m_RunOnPlatform = false;
            m_TestListGUIs[1].m_TestMode = TestMode.PlayMode;
            m_TestListGUIs[1].m_RunOnPlatform = false;
            m_TestListGUIs[2].m_TestMode = TestMode.PlayMode;
            m_TestListGUIs[2].m_RunOnPlatform = true;

            m_SelectedTestTypes = m_TestListGUIs[(int)testTypeToolbarIndex];
        }

        private void StartRetrieveTestList()
        {
            var listToInit = m_SelectedTestTypes;
            m_testRunnerApi.RetrieveTestList(listToInit.m_TestMode, rootTest =>
            {
                listToInit.Init(this, rootTest);
                listToInit.Reload();
            });
        }

        internal void OnGUI()
        {
            if (!m_Enabled)
            {
                Enable();
            }

            if (BuildPipeline.isBuildingPlayer)
            {
                m_IsBuilding = true;
            }
            else if (m_IsBuilding)
            {
                m_IsBuilding = false;
                Repaint();
            }

            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            var selectedIndex = m_TestTypeToolbarIndex;
            m_TestTypeToolbarIndex = (TestRunnerMenuLabels)GUILayout.Toolbar((int)m_TestTypeToolbarIndex, Enum.GetNames(typeof(TestRunnerMenuLabels)), "LargeButton", UnityEngine.GUI.ToolbarButtonSize.FitToContents);
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();

            if (selectedIndex != m_TestTypeToolbarIndex)
            {
                SelectTestListGUI(m_TestTypeToolbarIndex);
                StartRetrieveTestList();
            }

            EditorGUILayout.BeginVertical();
            using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
            {
                m_SelectedTestTypes.PrintHeadPanel();
            }
            EditorGUILayout.EndVertical();

            if (m_Settings.verticalSplit)
                SplitterGUILayout.BeginVerticalSplit(m_Spl);
            else
                SplitterGUILayout.BeginHorizontalSplit(m_Spl);

            EditorGUILayout.BeginVertical();
            EditorGUILayout.BeginVertical(Styles.testList);
            m_SelectedTestTypes.RenderTestList();
            EditorGUILayout.EndVertical();
            EditorGUILayout.EndVertical();

            m_SelectedTestTypes.RenderDetails(position.width);

            if (m_Settings.verticalSplit)
                SplitterGUILayout.EndVerticalSplit();
            else
                SplitterGUILayout.EndHorizontalSplit();

            m_SelectedTestTypes.PrintProgressBar(position);

            EditorGUILayout.BeginVertical();
            using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
            {
                m_SelectedTestTypes.PrintBottomPanel();
            }
            EditorGUILayout.EndVertical();
        }

        /// <summary>
        /// Adds additional menu items to the Test Runner window.
        /// </summary>
        /// <param name="menu">The <see cref="GenericMenu"/></param>
        public void AddItemsToMenu(GenericMenu menu)
        {
            menu.AddItem(m_GUIVerticalSplit, m_Settings.verticalSplit, m_Settings.ToggleVerticalSplit);
            menu.AddItem(m_GUIHorizontalSplit, !m_Settings.verticalSplit, m_Settings.ToggleVerticalSplit);

            menu.AddSeparator(null);

            if (EditorPrefs.GetBool("InternalMode", false))
            {
                menu.AddItem(m_GUIRunPlayModeTestAsEditModeTests, PlayerSettings.runPlayModeTestAsEditModeTest, () =>
                {
                    PlayerSettings.runPlayModeTestAsEditModeTest = !PlayerSettings.runPlayModeTestAsEditModeTest;
                });
            }

            if (PlayerSettings.playModeTestRunnerEnabled)
            {
                PlayerSettings.playModeTestRunnerEnabled = false;
                EditorUtility.DisplayDialog(m_GUIDisablePlaymodeTestsRunner.text, "You need to restart the editor now", "Ok");
            }
        }

        internal void RebuildUIFilter()
        {
            if (m_SelectedTestTypes != null && m_SelectedTestTypes.HasTreeData())
            {
                m_SelectedTestTypes.RebuildUIFilter();
            }
        }

        internal static void UpdateWindow()
        {
            if (s_Instance != null && s_Instance.m_SelectedTestTypes != null)
            {
                s_Instance.m_SelectedTestTypes.Repaint();
                s_Instance.Repaint();
            }
        }
    }
}
