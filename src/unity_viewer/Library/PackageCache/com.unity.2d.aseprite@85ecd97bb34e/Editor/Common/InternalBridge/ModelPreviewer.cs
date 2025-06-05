using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite.Common
{
    internal class ModelPreviewer : System.IDisposable
    {
        const float k_TimeControlRectHeight = 20;

        readonly PreviewRenderUtility m_RenderUtility;

        bool m_Disposed = false;
        Rect m_PreviewRect;
        Bounds m_RenderableBounds;
        Vector2Int m_ActorSize;

        TimeControl m_TimeControl;
        int m_Fps;
        Animator m_Animator;
        AnimationClip[] m_Clips;
        AnimationClip m_SelectedClip;
        List<float> m_FrameTimings;
        int m_ClipIndex = 0;
        SpriteRenderer[] m_Renderers;

        Texture m_Texture;
        GameObject m_PreviewObject;

        GUIContent[] m_ClipNames;
        int[] m_ClipIndices;

        public ModelPreviewer(GameObject assetPrefab, AnimationClip[] clips)
        {
            m_RenderUtility = new PreviewRenderUtility();
            m_RenderUtility.camera.fieldOfView = 30f;

            m_PreviewObject = m_RenderUtility.InstantiatePrefabInScene(assetPrefab);
            m_RenderUtility.AddManagedGameObject(m_PreviewObject);

            m_Renderers = m_PreviewObject.GetComponentsInChildren<SpriteRenderer>();
            m_RenderableBounds = GetRenderableBounds(m_Renderers);

            if (clips != null && clips.Length > 0)
            {
                SetupAnimation(clips);
                SelectClipFromIndex(m_ClipIndex);
            }
        }

        void SetupAnimation(AnimationClip[] clips)
        {
            m_TimeControl = new TimeControl();
            m_Animator = m_PreviewObject.GetComponent<Animator>();
            m_Clips = clips;

            var clipInfos = m_Animator.GetCurrentAnimatorClipInfo(0);

            var defaultClipName = string.Empty;
            if (clipInfos.Length > 0)
                defaultClipName = clipInfos[0].clip.name;

            m_ClipNames = new GUIContent[m_Clips.Length];
            m_ClipIndices = new int[m_Clips.Length];
            for (var i = 0; i < m_ClipNames.Length; ++i)
            {
                m_ClipNames[i] = new GUIContent(m_Clips[i].name);
                m_ClipIndices[i] = i;

                // Set starting clip to default clip.
                if (m_Clips[i].name == defaultClipName)
                    m_ClipIndex = i;
            }
        }

        void SelectClipFromIndex(int index)
        {
            m_SelectedClip = m_Clips[index];
            m_Fps = Mathf.RoundToInt(m_SelectedClip.frameRate);
            m_TimeControl.playbackSpeed = 1f / m_SelectedClip.length;
            m_TimeControl.currentTime = 0f;

            var timeSet = new HashSet<float>();
            var curveBindings = AnimationUtility.GetObjectReferenceCurveBindings(m_SelectedClip);

            for (var i = 0; i < curveBindings.Length; ++i)
            {
                var keyFrames = AnimationUtility.GetObjectReferenceCurve(m_SelectedClip, curveBindings[i]);
                for (var m = 0; m < keyFrames.Length; ++m)
                    timeSet.Add(keyFrames[m].time);
            }

            m_FrameTimings = new List<float>(timeSet.Count);
            
            // If the clip is empty, early out.
            if (timeSet.Count == 0)
                return;
            
            foreach (var time in timeSet)
                m_FrameTimings.Add(time);
            m_FrameTimings.Sort();
            // Remove the final frame time, as we add it on generation
            m_FrameTimings.RemoveAt(m_FrameTimings.Count - 1);
        }

        public void DrawPreview(Rect r, GUIStyle background)
        {
            if (!ShaderUtil.hardwareSupportsRectRenderTexture)
                return;
            var isRepainting = (Event.current.type == EventType.Repaint);

            if (isRepainting)
            {
                if (m_Texture != null)
                    Object.DestroyImmediate(m_Texture);
                m_Texture = null;
                m_PreviewRect = r;

                m_PreviewObject.transform.position = Vector3.zero;
                m_RenderUtility.BeginPreview(r, background);
                DoRenderPreview();
                m_Texture = m_RenderUtility.EndPreview();

                m_TimeControl?.Update();
            }

            if (m_SelectedClip != null)
                UpdateAnimation(isRepainting);
            UpdateActorSize();

            GUI.DrawTexture(r, m_Texture, ScaleMode.StretchToFill, false);

            if (m_SelectedClip != null)
                DrawTimeControlGUI(m_PreviewRect);
            else
                DrawInfoText(m_PreviewRect);
        }

        void UpdateAnimation(bool isRepainting)
        {
            if (!isRepainting || m_PreviewObject == null)
                return;

            m_TimeControl.loop = true;

            m_Animator.Play(m_SelectedClip.name, 0, m_TimeControl.normalizedTime);
            m_Animator.Update(m_TimeControl.deltaTime);
        }

        void UpdateActorSize()
        {
            if (m_Renderers == null || m_Renderers.Length == 0)
                return;

            var ppu = m_Renderers[0].sprite.pixelsPerUnit;
            var bounds = GetRenderableBounds(m_Renderers);

            m_ActorSize = new Vector2Int()
            {
                x = Mathf.RoundToInt(bounds.size.x * ppu),
                y = Mathf.RoundToInt(bounds.size.y * ppu)
            };
        }

        void DrawTimeControlGUI(Rect rect)
        {
            const float kSliderWidth = 150f;
            const float kSpacing = 4f;
            var timeControlRect = rect;

            // background
            GUI.Box(rect, GUIContent.none, EditorStyles.toolbar);

            timeControlRect.height = k_TimeControlRectHeight;
            timeControlRect.xMax -= kSliderWidth;

            var sliderControlRect = rect;
            sliderControlRect.height = k_TimeControlRectHeight;
            sliderControlRect.yMin += 1;
            sliderControlRect.yMax -= 1;
            sliderControlRect.xMin = sliderControlRect.xMax - kSliderWidth + kSpacing;

            m_TimeControl.DoTimeControl(timeControlRect);

            EditorGUI.BeginChangeCheck();
            m_ClipIndex = EditorGUI.IntPopup(sliderControlRect, m_ClipIndex, m_ClipNames, m_ClipIndices);
            if (EditorGUI.EndChangeCheck())
            {
                SelectClipFromIndex(m_ClipIndex);
            }

            DrawInfoText(rect);
        }

        void DrawInfoText(Rect rect)
        {
            rect.y = rect.yMax - 24;
            rect.height = 20;

            var text = "";
            if (m_TimeControl != null)
            {
                var currentTime = m_TimeControl.normalizedTime * m_SelectedClip.length;
                var currentFrame = GetFrameFromTime(currentTime);
                text += $"Frame {currentFrame} | ";
            }

            text += $"{m_ActorSize.x}x{m_ActorSize.y}";

            EditorGUI.DropShadowLabel(rect, text);
        }

        int GetFrameFromTime(float currentTime)
        {
            var frame = 0;
            for (var i = 0; i < m_FrameTimings.Count; ++i)
            {
                if (currentTime < m_FrameTimings[i])
                    break;
                frame++;
            }

            // Remove one to get the frame number start from 0
            return Mathf.Max(frame - 1, 0);
        }

        void DoRenderPreview()
        {
            var num1 = Mathf.Max(m_RenderableBounds.extents.magnitude, 0.0001f);
            var num2 = num1 * 3.8f;
            var vector3 = m_RenderableBounds.center - Quaternion.identity * (Vector3.forward * num2);
            m_RenderUtility.camera.transform.position = vector3;
            m_RenderUtility.camera.nearClipPlane = num2 - num1 * 1.1f;
            m_RenderUtility.camera.farClipPlane = num2 + num1 * 5.1f;
            m_RenderUtility.lights[0].intensity = 0.7f;
            m_RenderUtility.lights[1].intensity = 0.7f;
            m_RenderUtility.ambientColor = new Color(0.1f, 0.1f, 0.1f, 0.0f);

            m_RenderUtility.Render(true);
        }

        static Bounds GetRenderableBounds(SpriteRenderer[] renderers)
        {
            if (renderers.Length == 1)
            {
                var renderBound = renderers[0].bounds;
                var localPos = renderers[0].transform.localPosition;
                renderBound.center -= localPos;
                return renderBound;
            }

            var bounds = new Bounds();
            foreach (var rendererComponents in renderers)
            {
                var renderBound = rendererComponents.bounds;
                if (bounds.extents == Vector3.zero)
                    bounds = renderBound;
                else if (rendererComponents.enabled)
                    bounds.Encapsulate(renderBound);
            }
            return bounds;
        }

        public void Dispose()
        {
            if (m_Disposed)
                return;

            m_RenderUtility.Cleanup();
            Object.DestroyImmediate(m_PreviewObject);
            m_PreviewObject = null;
            m_Disposed = true;
        }
    }
}
