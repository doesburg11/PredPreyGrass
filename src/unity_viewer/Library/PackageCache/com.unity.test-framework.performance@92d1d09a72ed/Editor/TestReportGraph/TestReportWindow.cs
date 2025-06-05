using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;
using System;
using UnityEditor.IMGUI.Controls;
using System.IO;
using System.Linq;
using System.Text;
using Unity.PerformanceTesting.Data;
using Unity.PerformanceTesting.Editor.UIElements;
using Unity.PerformanceTesting.Runtime;

namespace Unity.PerformanceTesting.Editor
{
    internal class TestReportWindow : EditorWindow
    {
        static int s_windowWidth = 800;
        static int s_windowHeight = 600;
        private GUIStyle m_glStyle = null;
        private GUIStyle m_boldStyle = null;
        private Material m_material;
        public Color m_colorWhite = new Color(1.0f, 1.0f, 1.0f);
        public Color m_colorBarBackground = new Color(0.5f, 0.5f, 0.5f);
        public Color m_colorBoxAndWhiskerBackground = new Color(0.4f, 0.4f, 0.4f);
        public Color m_colorBar = new Color(0.95f, 0.95f, 0.95f);
        public Color m_colorStandardLine = new Color(1.0f, 1.0f, 1.0f);
        public Color m_colorMedianLine = new Color(0.2f, 0.5f, 1.0f, 0.5f);
        public Color m_colorMedianText = new Color(0.4f, 0.7f, 1.0f, 1.0f);
        public Color m_colorWarningText = Color.red;
        public Color m_toolbarSeparator = new Color (0.15f, 0.15f, 0.15f, 1.0f);
        private Run m_resultsData = null;
        private string m_selectedTest;

        private DateTime m_lastResultsDateTime = new DateTime(0);
        private bool m_NewerFileExists = false;
        private double m_LastFileCheckTime = 0f;
        private float m_FileCheckFrequencySeconds = 3f;
        private bool m_AutoRefresh;

        private List<string> m_sampleGroups = new List<string>();

        //private int m_selectedSampleGroupIndex;
        private bool m_showTests = true;
        private bool m_showSamples = true;
        private int[] m_columnWidth = new int[4];

        private bool m_isResizing;
        private float m_testListHeight = (float)s_windowHeight / 4;
        private Rect m_splitterRect;
        private float m_windowHeight;

        [SerializeField]
        TreeViewState m_testListTreeViewState;
        [SerializeField]
        MultiColumnHeaderState m_testListMulticolumnHeaderState;
        TestListTable m_testListTable;

        Vector2 m_sampleGroupScroll = new Vector2(0, 0);
        private readonly Dictionary<Tuple<string, string>, SampleGroupAdditionalData> m_sampleGroupAdditionalData = new Dictionary<Tuple<string, string>, SampleGroupAdditionalData>();
        private bool m_isResultsCountPositive;
        private ToolbarWithSearch m_toolbarWithSearch;
        public string searchString;

        private const int m_chartLimit = 1000;

        private void CreateTestListTable()
        {
            if (m_testListTreeViewState == null)
                m_testListTreeViewState = new TreeViewState();

            //if (m_profileMulticolumnHeaderState==null)
            m_testListMulticolumnHeaderState = TestListTable.CreateDefaultMultiColumnHeaderState(700);

            var multiColumnHeader = new MultiColumnHeader(m_testListMulticolumnHeaderState);
            multiColumnHeader.SetSorting((int)TestListTable.MyColumns.Name, true);
            multiColumnHeader.ResizeToFit();
            m_testListTable = new TestListTable(m_testListTreeViewState, multiColumnHeader, this);
            m_testListTable.ExpandAll();
        }

        public Run GetResults()
        {
            return m_resultsData;
        }

        public void SelectTest(int index)
        {
            if (index < 0 || index >= m_resultsData.Results.Count)
            {
                m_sampleGroups.Clear();
                return;
            }

            var result = m_resultsData.Results[index];
            SelectTest(result);
        }

        public void SelectTest(string name)
        {
            foreach (var result in m_resultsData.Results)
            {
                if (result.Name == name)
                {
                    SelectTest(result);
                    return;
                }
            }
        }

        public void SelectTest(PerformanceTestResult result)
        {
            m_selectedTest = result.Name;

            m_sampleGroups.Clear();
            foreach (var sampleGroup in result.SampleGroups)
            {
                m_sampleGroups.Add(sampleGroup.Name);
            }
        }

        [MenuItem("Window/General/Performance Test Report", false, priority = 202)]
        private static void Init()
        {
            var window = GetWindow<TestReportWindow>("Test Report");
            window.minSize = new Vector2(640, 480);
            window.position.size.Set(s_windowWidth, s_windowHeight);
            window.Show();
        }

        public void SetupMaterial()
        {
            // Create a new material
            m_material = new Material(Shader.Find("Unlit/TestReportShader"));
            m_material.SetPass(0);
        }

        private string GetResultsPath()
        {
            return Path.Combine(Application.persistentDataPath, "PerformanceTestResults.json");
        }

        private bool NewerFileExists()
        {
            m_LastFileCheckTime = EditorApplication.timeSinceStartup;

            string filePath = GetResultsPath();
            if (!File.Exists(filePath))
                return false;

            if (m_resultsData == null)
                return true;

            DateTime dateTime = File.GetLastWriteTime(GetResultsPath());
            if ((dateTime - m_lastResultsDateTime).TotalMilliseconds > 0)
                return true;

            return false;
        }

        private void ResetFileCheck()
        {
            m_lastResultsDateTime = File.GetLastWriteTime(GetResultsPath());
            m_NewerFileExists = false;
            m_LastFileCheckTime = EditorApplication.timeSinceStartup;
        }

        private bool CheckIfNewerFileExists()
        {
            if (EditorApplication.timeSinceStartup - m_LastFileCheckTime < m_FileCheckFrequencySeconds)
                return m_NewerFileExists;

            m_NewerFileExists = NewerFileExists();
            return m_NewerFileExists;
        }

        private void LoadData()
        {
            m_toolbarWithSearch?.ClearSearchString();
            string filePath = GetResultsPath();
            if (!File.Exists(filePath)) return;

            string json = File.ReadAllText(filePath);
            m_resultsData = JsonUtility.FromJson<Run>(json);
            ResetFileCheck();

            List<SamplePoint> samplePoints = new List<SamplePoint>();

            m_sampleGroupAdditionalData.Clear();
            foreach (var result in m_resultsData.Results)
            {
                foreach (var sampleGroup in result.SampleGroups)
                {
                    samplePoints.Clear();
                    for (int index = 0; index < sampleGroup.Samples.Count; index++)
                    {
                        var sample = sampleGroup.Samples[index];
                        samplePoints.Add(new SamplePoint(sample, index));
                    }

                    samplePoints.Sort();

                    int discard;
                    SampleGroupAdditionalData data = new SampleGroupAdditionalData();
                    data.min = (float)GetPercentageOffset(samplePoints, 0, out discard);
                    data.lowerQuartile = (float)GetPercentageOffset(samplePoints, 25, out discard);
                    data.median = (float)sampleGroup.Median;
                    data.upperQuartile = (float)GetPercentageOffset(samplePoints, 75, out discard);
                    data.max = (float)GetPercentageOffset(samplePoints, 100, out discard);

                    var key = Tuple.Create(result.Name, sampleGroup.Name);
                    m_sampleGroupAdditionalData[key] = data;
                }
            }

            CreateTestListTable();
        }

        private void OnEnable()
        {
            SetupMaterial();
            LoadData();
        }

        private void OnFocus()
        {
            CheckIfNewerFileExists();
        }

        private void OnGUI()
        {
            if (m_glStyle == null)
            {
                m_glStyle = new GUIStyle(GUI.skin.box);
                m_glStyle.padding = new RectOffset(0, 0, 0, 0);
                m_glStyle.margin = new RectOffset(0, 0, 0, 0);
            }

            if (m_boldStyle == null)
            {
                m_boldStyle = new GUIStyle(GUI.skin.label);
                m_boldStyle.fontStyle = FontStyle.Bold;
            }

            if (m_toolbarWithSearch == null)
            {
                m_toolbarWithSearch = new ToolbarWithSearch();
                m_toolbarWithSearch.SearchTextChanged += s =>
                {
                    searchString = s;
                    CreateTestListTable();
                };
            }

            m_isResultsCountPositive = m_resultsData?.Results.Count > 0;
            DrawToolbar();

            if (!DrawPerformanceDataStatusLabel()) return;

            m_toolbarWithSearch.Draw();
            DrawTestView();
            DrawSampleView();
        }

        private void Update()
        {
            CheckIfNewerFileExists();
            if (m_NewerFileExists && m_AutoRefresh)
                Refresh();
        }

        private double GetPercentageOffset(List<SamplePoint> samplePoint, float percent, out int outputFrameIndex)
        {
            int index = (int)((samplePoint.Count - 1) * percent / 100);
            outputFrameIndex = samplePoint[index].index;

            // True median is half of the sum of the middle 2 frames for an even count. However this would be a value never recorded so we avoid that.
            return samplePoint[index].sample;
        }

        private bool BoldFoldout(bool toggle, string text)
        {
            GUIStyle foldoutStyle = new GUIStyle(EditorStyles.foldout);
            foldoutStyle.fontStyle = FontStyle.Bold;
            return EditorGUILayout.Foldout(toggle, text, foldoutStyle);

            /*
            EditorGUILayout.LabelField(text, EditorStyles.boldLabel, GUILayout.Width(100));
            return true;
            */
        }

        private void SetColumnSizes(int a, int b, int c, int d)
        {
            m_columnWidth[0] = a;
            m_columnWidth[1] = b;
            m_columnWidth[2] = c;
            m_columnWidth[3] = d;
        }

        private void DrawColumn(int n, string col)
        {
            if (m_columnWidth[n] > 0)
                EditorGUILayout.LabelField(col, GUILayout.Width(m_columnWidth[n]));
            else
                EditorGUILayout.LabelField(col);
        }

        private void DrawColumn(int n, float value)
        {
            DrawColumn(n, string.Format("{0:f2}", value));
        }

        private void Draw2Column(string label, float value)
        {
            EditorGUILayout.BeginHorizontal();
            DrawColumn(0, label);
            DrawColumn(1, value);
            EditorGUILayout.EndHorizontal();
        }

        private void Refresh()
        {
            LoadData();
            CreateTestListTable();
            if (m_resultsData == null) return;
            if (m_resultsData.Results == null) return;
            if (m_resultsData.Results.Any(result => result.Name == m_selectedTest))
                SelectTest(m_selectedTest);
            else
                SelectTest(0);
            Repaint();
        }

        private void Export()
        {
            if (m_resultsData == null || m_resultsData.Results.Count <= 0) return;

            var path = EditorUtility.SaveFilePanel("Save CSV data", "", "PerformanceTestResults.csv", "csv");
            if (string.IsNullOrEmpty(path)) return;

            var sb = new StringBuilder();
            const string header =
                "Index,Test Name,Version,Sample Group Name,Unit,Increase Is Better,Min,Max,Median,Average,Standard Deviation,Sum,Values\n";
            sb.Append(header);

            var index = 0;
            foreach (var result in m_resultsData.Results)
            foreach (var sampleGroup in result.SampleGroups)
            {
                var sampleValues = string.Join(", ", sampleGroup.Samples);
                var increaseIsBetter = sampleGroup.IncreaseIsBetter ? "Yes" : "No";
                var line =
                    $"{index},\"{result.Name}\",{result.Version},\"{sampleGroup.Name}\",\"{sampleGroup.Unit}\",{increaseIsBetter},{sampleGroup.Min},{sampleGroup.Max},{sampleGroup.Median},{sampleGroup.Average},{sampleGroup.StandardDeviation},{sampleGroup.Sum},{sampleValues}\n";
                sb.Append(line);
                index++;
            }

            File.WriteAllText(path, sb.ToString());
        }
        
        private void ClearResults()
        {
            File.Delete(GetResultsPath());
            m_resultsData = null;
            GUILayout.Label(string.Empty);
        }

        private void DrawToolbar()
        {
            GUILayout.BeginHorizontal();
            m_AutoRefresh = EditorPrefs.GetBool("ToggleState", false);
            m_AutoRefresh = GUILayout.Toggle(m_AutoRefresh, "Auto Refresh");
            EditorPrefs.SetBool("ToggleState", m_AutoRefresh);

            if (GUILayout.Button("Refresh", EditorStyles.toolbarButton)) Refresh();

            GUILayout.FlexibleSpace();

            if (m_isResultsCountPositive) GUILayout.Label($"Last results {m_lastResultsDateTime}");

            GUI.enabled = m_isResultsCountPositive;
            if (GUILayout.Button("Clear Results", EditorStyles.toolbarButton))
            {
                if (EditorUtility.DisplayDialog("Clear Results",
                        "Are you sure you want to clear all Performance test results?", "Clear", "Cancel"))
                {
                    ClearResults();
                }
            }

            if (GUILayout.Button("Export", EditorStyles.toolbarButton)) Export();
            GUI.enabled = true;

            DrawHelpIcon();
            GUILayout.EndHorizontal();

            EditorGUI.DrawRect(new Rect(0, GUILayoutUtility.GetLastRect().yMax, position.width, 1), m_toolbarSeparator);
            EditorGUILayout.Space();
        }

        private static void DrawHelpIcon()
        {
            if (!GUILayout.Button(EditorGUIUtility.TrIconContent("_Help", "Open Documentation"),
                    EditorStyles.toolbarButton)) return;
            var packageInfo = UnityEditor.PackageManager.PackageInfo.FindForAssetPath("Packages/com.unity.test-framework.performance");
            var shortVersion = packageInfo.version.Substring(0,
                packageInfo.version.IndexOf('.', packageInfo.version.IndexOf('.') + 1));
            var documentationUrl = $"https://docs.unity3d.com/Packages/{packageInfo.name}@{shortVersion}";
            Help.ShowHelpPage(documentationUrl);
        }

        private bool DrawPerformanceDataStatusLabel()
        {
            var isDataMissing = m_resultsData == null || !m_isResultsCountPositive;
            var shouldRefreshData = m_NewerFileExists && !m_AutoRefresh;

            if (isDataMissing || shouldRefreshData)
            {
                var content = new GUIContent
                {
                    image = EditorGUIUtility.IconContent("console.infoicon").image,
                    text = shouldRefreshData
                        ? "New performance <b>data is available</b>. Click <b>\"Refresh\"</b> or <b>\"Auto Refresh\"</b> to view it"
                        : "No performance data found"
                };
                
                var style = new GUIStyle(GUI.skin.GetStyle("HelpBox"))
                {
                    richText = true,
                    alignment = TextAnchor.MiddleLeft
                };

                GUILayout.Label(content, style);

                if (isDataMissing) return false;
            }

            return true;
        }

        private void DrawTestView()
        {
            m_showTests = BoldFoldout(m_showTests, "Test View");
            if (m_showTests)
            {
                if (m_testListTable != null)
                {
                    var r = GUILayoutUtility.GetRect(position.width, m_testListHeight, GUI.skin.box,
                        GUILayout.ExpandWidth(true));
                    m_testListTable.OnGUI(r);
                    Resize(r.y);
                }
            }

            if (!string.IsNullOrEmpty(m_selectedTest))
            {
                var profileDataFile = Path.Combine(Application.persistentDataPath,
                    Utils.RemoveIllegalCharacters(m_selectedTest) + ".raw");
                
                if (File.Exists(profileDataFile))
                {
                    if (GUILayout.Button($"Load profiler data for test: {m_selectedTest}"))
                    {
                        ProfilerDriver.LoadProfile(profileDataFile, false);
                    }
                }
            }
        }

        private void DrawSampleView()
        {
            m_showSamples = BoldFoldout(m_showSamples, "Sample Group View");

            if (m_showSamples)
            {
                if (m_sampleGroups?.Count <= 0)
                {
                    EditorGUILayout.LabelField("This Selection Has No Sample Groups", m_boldStyle);
                    return;
                }

                SetColumnSizes(50, 50, 50, 50);

                var boxStyle = GUI.skin.box;
                var boxWidth = position.width - GUI.skin.verticalScrollbar.fixedWidth -
                                (boxStyle.padding.horizontal + boxStyle.margin.horizontal);
                var graphWidth = position.width - 200;

                EditorGUILayout.BeginVertical();
                m_sampleGroupScroll = EditorGUILayout.BeginScrollView(m_sampleGroupScroll, false, true);

                foreach (var result in m_resultsData.Results)
                {
                    if (result.Name != m_selectedTest) continue;

                    EditorGUILayout.LabelField(result.Name, m_boldStyle);

                    foreach (var sampleGroup in result.SampleGroups)
                    {
                        var key = Tuple.Create(result.Name, sampleGroup.Name);

                        if (m_sampleGroupAdditionalData.TryGetValue(key, out var data))
                        {
                            var samplesToDraw = sampleGroup.Samples;
                            var min = data.min;
                            var lowerQuartile = data.lowerQuartile;
                            var median = data.median;
                            var upperQuartile = data.upperQuartile;
                            var max = data.max;
                            var graphMin = min > 0.0f ? 0.0f : min;

                            EditorGUILayout.BeginVertical(boxStyle, GUILayout.Width(boxWidth),
                                GUILayout.ExpandHeight(false));
                            EditorGUILayout.LabelField(sampleGroup.Name, m_boldStyle);
                            EditorGUILayout.LabelField($"Sample Unit: {sampleGroup.Unit}");

                            if (samplesToDraw.Count > m_chartLimit)
                            {
                                EditorGUILayout.HelpBox(
                                    $"Sample Group has more than {m_chartLimit} Samples. The first {m_chartLimit} Samples will be displayed. However, calculations are done according to all samples received from the test run.",
                                    MessageType.Warning, true);
                                samplesToDraw = samplesToDraw.Take(m_chartLimit).ToList();
                            }

                            EditorGUILayout.BeginHorizontal(GUILayout.Height(100), GUILayout.ExpandHeight(false));
                            EditorGUILayout.BeginVertical(GUILayout.Width(100), GUILayout.ExpandHeight(true));
                            Draw2Column("Max", max);
                            GUILayout.FlexibleSpace();
                            var oldColor = GUI.contentColor;
                            GUI.contentColor = median < 0.01f ? m_colorWarningText : m_colorMedianText;
                            Draw2Column("Median", median);
                            GUI.contentColor = oldColor;
                            GUILayout.FlexibleSpace();
                            Draw2Column("Min", min);
                            EditorGUILayout.EndVertical();
                            DrawBarGraph(graphWidth, 100, samplesToDraw, graphMin, max, median);
                            DrawBoxAndWhiskerPlot(50, 100, min, lowerQuartile, median, upperQuartile, max, min, max,
                                (float)sampleGroup.StandardDeviation, m_colorWhite, m_colorBoxAndWhiskerBackground);
                            EditorGUILayout.EndHorizontal();
                            EditorGUILayout.EndVertical();
                        }
                    }
                }
                EditorGUILayout.EndScrollView();
                EditorGUILayout.EndVertical();
            }
        }

        private void Resize(float headerHeight)
        {
            m_splitterRect = new Rect(0, headerHeight + m_testListHeight, position.width, 5f);

            EditorGUIUtility.AddCursorRect(m_splitterRect, MouseCursor.ResizeVertical);

            if (Event.current.type == EventType.MouseDown)
                m_isResizing = m_splitterRect.Contains(Event.current.mousePosition);

            const float minListHeight = 80f;
            var maxListHeight = position.height - 240f;
            var needMoveSplitterAndRepaint = false;

            if (m_isResizing && Event.current.type == EventType.MouseDrag)
            {
                m_testListHeight = Mathf.Clamp(Event.current.mousePosition.y - headerHeight, minListHeight, maxListHeight);
                needMoveSplitterAndRepaint = true;
            }
            else if (Math.Abs(m_windowHeight - position.height) > float.Epsilon)
            {
                m_windowHeight = position.height;
                m_testListHeight = Mathf.Clamp(m_testListHeight, minListHeight, maxListHeight);
                needMoveSplitterAndRepaint = true;
            }

            if (needMoveSplitterAndRepaint)
            {
                m_splitterRect.Set(m_splitterRect.x, m_testListHeight, m_splitterRect.width, m_splitterRect.height);
                Repaint();
            }

            if (Event.current.type == EventType.MouseUp)
                m_isResizing = false;
        }

        private void DrawBarGraph(float width, float height, List<double> samples, float min, float max, float median)
        {
            if (DrawStart(width, height))
            {
                Rect rect = GUILayoutUtility.GetLastRect();

                float clipH = Math.Max(m_sampleGroupScroll.y - rect.y, 0);
                float maxH = Math.Max(height - clipH, 0);

                int xAxisDivisions = samples.Count;
                float xAxisInc = width / xAxisDivisions;

                float yRange = max - min;
                float x = 0;
                float y = 0;
                float spacing = 2;
                float w = xAxisInc - spacing;
                if (w < 1) spacing = 0;

                float h = 0;
                for (int i = 0; i < samples.Count; i++)
                {
                    float sample = (float)samples[i];

                    w = xAxisInc - spacing;
                    h = ((sample - min) * height) / yRange;
                    if (h > maxH)
                        h = maxH;
                    DrawBar(x, y + (height - h), w, h, m_colorBar);

                    x += xAxisInc;
                }

                h = ((median - min) * height) / yRange;
                if (h <= maxH)
                {
                    //DrawLine(0, (height - h), width, (height - h), m_colorMedianLine);
                    DrawBar(0, (height - h), width, 3, m_colorMedianLine);
                }

                x = 0;
                for (int i = 0; i < samples.Count; i++)
                {
                    float sample = (float)samples[i];

                    string tooltip = string.Format("{0} (at sample {1} of {2})", sample, i+1, samples.Count);

                    GUI.Label(new Rect(rect.x + x, rect.y + y, xAxisInc, height), new GUIContent("", tooltip));

                    x += xAxisInc;
                }

                DrawEnd();
            }
        }

        public bool DrawStart(Rect r)
        {
            if (Event.current.type != EventType.Repaint)
                return false;

            GL.PushMatrix();
            SetupMaterial();

            Matrix4x4 matrix = new Matrix4x4();
            matrix.SetTRS(new Vector3(r.x, r.y, 0), Quaternion.identity, Vector3.one);
            GL.MultMatrix(matrix);
            return true;
        }

        public bool DrawStart(float w, float h, GUIStyle style = null)
        {
            Rect r = GUILayoutUtility.GetRect(w, h, style == null ? m_glStyle : style);
            return DrawStart(r);
        }

        public void DrawEnd()
        {
            GL.PopMatrix();
        }

        public void DrawBar(float x, float y, float w, float h, Color col)
        {
            GL.Begin(GL.TRIANGLE_STRIP);
            GL.Color(col);
            GL.Vertex3(x, y, 0);
            GL.Vertex3(x + w, y, 0);
            GL.Vertex3(x, y + h, 0);
            GL.Vertex3(x + w, y + h, 0);
            GL.End();
        }

        void DrawBar(float x, float y, float w, float h, float r, float g, float b)
        {
            DrawBar(x, y, w, h, new Color(r, g, b));
        }

        void DrawLine(float x, float y, float x2, float y2, Color col)
        {
            GL.Begin(GL.LINES);
            GL.Color(col);
            GL.Vertex3(x, y, 0);
            GL.Vertex3(x2, y2, 0);
            GL.End();
        }

        void DrawLine(float x, float y, float x2, float y2, float r, float g, float b)
        {
            DrawLine(x, y, x2, y2, new Color(r, g, b));
        }

        void DrawBox(float x, float y, float w, float h, Color col)
        {
            GL.Begin(GL.LINE_STRIP);
            GL.Color(col);
            GL.Vertex3(x, y, 0);
            GL.Vertex3(x + w, y, 0);
            GL.Vertex3(x + w, y + h, 0);
            GL.Vertex3(x, y + h, 0);
            GL.Vertex3(x, y, 0);
            GL.End();
        }

        void DrawBox(float x, float y, float w, float h, float r, float g, float b)
        {
            DrawBox(x, y, w, h, new Color(r, g, b));
        }

        private float ClampToRange(float value, float min, float max)
        {
            return Math.Max(min, Math.Min(value, max));
        }

        private void DrawHistogramStart(float width)
        {
            EditorGUILayout.BeginHorizontal(GUILayout.Width(width + 10));

            //GUILayoutUtility.GetRect(GUI.skin.box.margin.left, 1);

            EditorGUILayout.BeginVertical();
        }

        private void DrawHistogramEnd(float width, float min, float max, float spacing)
        {
            EditorGUILayout.BeginHorizontal();
            float lastBar = width - 50;
            GUIStyle rightAlignStyle = new GUIStyle(GUI.skin.label);
            rightAlignStyle.alignment = TextAnchor.MiddleCenter;
            EditorGUILayout.LabelField(string.Format("{0:f2}", min), GUILayout.Width(lastBar));
            EditorGUILayout.LabelField(string.Format("{0:f2}", max), rightAlignStyle, GUILayout.Width(50));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.EndVertical();

            EditorGUILayout.EndHorizontal();
        }

        private void DrawHistogramBackground(float width, float height, int bucketCount, float spacing)
        {
            float x = (spacing / 2);
            float y = 0;
            float w = ((width + spacing) / bucketCount) - spacing;
            float h = height;

            for (int i = 0; i < bucketCount; i++)
            {
                DrawBar(x, y, w, h, m_colorBarBackground);
                x += w;
                x += spacing;
            }
        }

        private void DrawHistogramData(float width, float height, int[] buckets, int totalFrameCount, float min,
            float max, Color barColor, float spacing)
        {
            float x = (spacing / 2);
            float y = 0;
            float w = ((width + spacing) / buckets.Length) - spacing;
            float h = height;

            Rect rect = GUILayoutUtility.GetLastRect();

            int bucketCount = buckets.Length;
            float bucketWidth = ((max - min) / bucketCount);
            for (int bucketAt = 0; bucketAt < bucketCount; bucketAt++)
            {
                var count = buckets[bucketAt];

                float barHeight = (h * count) / totalFrameCount;
                if (barHeight > rect.height)
                    barHeight = rect.height;
                DrawBar(x, y + (h - barHeight), w, barHeight, barColor);

                float bucketStart = min + (bucketAt * bucketWidth);
                float bucketEnd = bucketStart + bucketWidth;
                GUI.Label(new Rect(rect.x + x, rect.y + y, w, h),
                    new GUIContent("", string.Format("{0:f2}-{1:f2}ms\n{2} frames", bucketStart, bucketEnd, count)));

                x += w;
                x += spacing;
            }
        }

        private void DrawHistogram(float width, float height, int[] buckets, int totalFrameCount, float min, float max,
            Color barColor)
        {
            DrawHistogramStart(width);

            float spacing = 2;

            if (DrawStart(width, height))
            {
                Rect rect = GUILayoutUtility.GetLastRect();

                float clipH = Math.Max(m_sampleGroupScroll.y - rect.y, 0);
                float maxH = Math.Max(height - clipH, 0);

                DrawHistogramBackground(width, maxH, buckets.Length, spacing);
                DrawHistogramData(width, height, buckets, totalFrameCount, min, max, barColor, spacing);
                DrawEnd();
            }

            DrawHistogramEnd(width, min, max, spacing);
        }

        private void DrawBoxAndWhiskerPlot(float width, float height, float min, float lowerQuartile, float median,
            float upperQuartile, float max, float yAxisStart, float yAxisEnd, float standardDeviation, Color color,
            Color colorBackground)
        {
            if (DrawStart(width, height))
            {
                Rect rect = GUILayoutUtility.GetLastRect();
                float clipH = Math.Max(m_sampleGroupScroll.y - rect.y, 0);
                float maxH = Math.Max(height - clipH, 0);
                if (maxH > 0)
                {
                    float x = 0;
                    float y = 0;
                    float w = width;
                    float h = height;

                    DrawBoxAndWhiskerPlot(rect, x, y, w, h, min, lowerQuartile, median, upperQuartile, max, yAxisStart,
                        yAxisEnd, standardDeviation, color, colorBackground);
                }

                DrawEnd();
            }
        }

        private void DrawBoxAndWhiskerPlot(Rect rect, float x, float y, float w, float h, float min,
            float lowerQuartile, float median, float upperQuartile, float max, float yAxisStart, float yAxisEnd,
            float standardDeviation, Color color, Color colorBackground)
        {
            string tooltip = string.Format(
                "Max :\t\t{0:f2}\n\nUpper Quartile :\t{1:f2}\nMedian :\t\t{2:f2}\nLower Quartile :\t{3:f2}\n\nMin :\t\t{4:f2}\n\nStandard Deviation:\t{5:f2}",
                max, upperQuartile, median, lowerQuartile, min,
                standardDeviation);
            GUI.Label(rect, new GUIContent("", tooltip));

            float clipH = Math.Max(m_sampleGroupScroll.y - rect.y, 0);
            float maxH = Math.Max(h - clipH, 0);

            DrawBar(x, y + (h - maxH), w, maxH, m_colorBoxAndWhiskerBackground);

            float first = yAxisStart;
            float last = yAxisEnd;
            float range = last - first;

            bool startCap = (min >= first) ? true : false;
            bool endCap = (max <= last) ? true : false;

            // Range clamping
            min = ClampToRange(min, first, last);
            lowerQuartile = ClampToRange(lowerQuartile, first, last);
            median = ClampToRange(median, first, last);
            upperQuartile = ClampToRange(upperQuartile, first, last);
            max = ClampToRange(max, first, last);

            float yMin = h * (min - first) / range;
            float yLowerQuartile = h * (lowerQuartile - first) / range;
            float yMedian = h * (median - first) / range;
            float yUpperQuartile = h * (upperQuartile - first) / range;
            float yMax = h * (max - first) / range;

            // Clamp to scroll area
            yMin = Math.Min(yMin, maxH);
            yLowerQuartile = Math.Min(yLowerQuartile, maxH);
            yMedian = Math.Min(yMedian, maxH);
            yUpperQuartile = Math.Min(yUpperQuartile, maxH);
            yMax = Math.Min(yMax, maxH);

            // Min to max line
            //DrawLine(x + (w / 2), y + (h - yMin), x + (w / 2), y + (h - yMax), color);
            DrawLine(x + (w / 2), y + (h - yMin), x + (w / 2), y + (h - yLowerQuartile), color);
            DrawLine(x + (w / 2), y + (h - yUpperQuartile), x + (w / 2), y + (h - yMax), color);

            // Quartile boxes
            float xMargin = (2 * w / 8);
            if (colorBackground != color)
                DrawBar(x + xMargin, y + (h - yMedian), w - (2 * xMargin), (yMedian - yLowerQuartile), colorBackground);
            DrawBox(x + xMargin, y + (h - yMedian), w - (2 * xMargin), (yMedian - yLowerQuartile), color);
            if (colorBackground != color)
                DrawBar(x + xMargin, y + (h - yUpperQuartile), w - (2 * xMargin), (yUpperQuartile - yMedian),
                    colorBackground);
            DrawBox(x + xMargin, y + (h - yUpperQuartile), w - (2 * xMargin), (yUpperQuartile - yMedian), color);

            // Median line
            DrawLine(x + xMargin, y + (h - yMedian), x + (w - xMargin), y + (h - yMedian), color);

            // Line caps
            xMargin = (3 * w / 8);
            if (startCap)
                DrawLine(x + xMargin, y + (h - yMin), x + (w - xMargin), y + (h - yMin), color);

            if (endCap)
                DrawLine(x + xMargin, y + (h - yMax), x + (w - xMargin), y + (h - yMax), color);
        }
    }
}
