using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class SkeletonView : ISkeletonView
    {
        internal const string deleteCommandName = "Delete";
        internal const string softDeleteCommandName = "SoftDelete";

        const float k_PickingRadius = 5f;
        static readonly int k_BodyHashCode = "Body".GetHashCode();
        static readonly int k_JointHashCode = "Joint".GetHashCode();
        static readonly int k_TailHashCode = "Tail".GetHashCode();
        static readonly int k_CreateBoneHashCode = "CreateBone".GetHashCode();

        public int InvalidID { get; set; }
        public SkeletonMode mode { get; set; }
        public int defaultControlID { get; set; }
        public int hoveredBoneID => m_HoveredBoneID;
        public int hoveredJointID => m_HoveredJointID;
        public int hoveredBodyID => m_HoveredBodyID;
        public int hoveredTailID => m_HoveredTailID;
        public int hotBoneID => m_HotBoneID;

        IGUIWrapper m_GUIWrapper;
        int m_RotateControlID = -1;
        int m_MoveControlID = -1;
        int m_FreeMoveControlID = -1;
        int m_MoveJointControlID = -1;
        int m_MoveEndPositionControlID = -1;
        int m_ChangeLengthControlID = -1;
        int m_CreateBoneControlID = -1;
        int m_HoveredBoneID = 0;
        int m_PrevHoveredBoneID = 0;
        int m_HoveredBodyID = 0;
        int m_HoveredJointID = 0;
        int m_HoveredTailID = 0;
        int m_HotBoneID = 0;
        int m_HoveredBodyControlID = -1;
        int m_HoveredJointControlID = -1;
        int m_HoveredTailControlID = -1;
        float m_NearestDistance;
        float m_NearestBodyDistance;
        float m_NearestJointDistance;
        float m_NearestTailDistance;
        int m_NearestBodyId = 0;
        int m_NearestJointId = 0;
        int m_NearestTailId = 0;
        SliderData m_HoveredSliderData = SliderData.zero;
        SliderData m_HotSliderData = SliderData.zero;

        public SkeletonView(IGUIWrapper gw)
        {
            m_GUIWrapper = gw;
        }

        public void BeginLayout()
        {
            m_HoveredBodyControlID = m_GUIWrapper.GetControlID(k_BodyHashCode, FocusType.Passive);
            m_HoveredJointControlID = m_GUIWrapper.GetControlID(k_JointHashCode, FocusType.Passive);
            m_HoveredTailControlID = m_GUIWrapper.GetControlID(k_TailHashCode, FocusType.Passive);
            m_CreateBoneControlID = m_GUIWrapper.GetControlID(k_CreateBoneHashCode, FocusType.Passive);

            if (m_GUIWrapper.eventType == EventType.Layout)
            {
                m_PrevHoveredBoneID = m_HoveredBoneID;
                m_NearestDistance = float.MaxValue;
                m_NearestBodyDistance = float.MaxValue;
                m_NearestJointDistance = float.MaxValue;
                m_NearestTailDistance = float.MaxValue;
                m_NearestBodyId = InvalidID;
                m_NearestJointId = InvalidID;
                m_NearestTailId = InvalidID;
                m_HoveredBoneID = InvalidID;
                m_HoveredBodyID = InvalidID;
                m_HoveredJointID = InvalidID;
                m_HoveredTailID = InvalidID;
                m_HoveredSliderData = SliderData.zero;

                if (m_GUIWrapper.IsControlHot(0))
                {
                    m_RotateControlID = -1;
                    m_MoveControlID = -1;
                    m_FreeMoveControlID = -1;
                    m_MoveJointControlID = -1;
                    m_MoveEndPositionControlID = -1;
                    m_ChangeLengthControlID = -1;
                    m_HotBoneID = InvalidID;
                }
            }
        }

        public void EndLayout()
        {
            m_GUIWrapper.LayoutControl(m_HoveredBodyControlID, m_NearestBodyDistance * 0.25f);
            m_GUIWrapper.LayoutControl(m_HoveredJointControlID, m_NearestJointDistance);
            m_GUIWrapper.LayoutControl(m_HoveredTailControlID, m_NearestTailDistance);

            if (m_GUIWrapper.IsControlNearest(m_HoveredBodyControlID))
            {
                m_HoveredBoneID = m_NearestBodyId;
                m_HoveredBodyID = m_NearestBodyId;
            }

            if (m_GUIWrapper.IsControlNearest(m_HoveredJointControlID))
            {
                m_HoveredBoneID = m_NearestJointId;
                m_HoveredJointID = m_NearestJointId;
            }

            if (m_GUIWrapper.IsControlNearest(m_HoveredTailControlID))
            {
                m_HoveredBoneID = m_NearestTailId;
                m_HoveredTailID = m_NearestTailId;
            }

            if ((m_GUIWrapper.eventType == EventType.Layout && m_PrevHoveredBoneID != m_HoveredBoneID) || m_GUIWrapper.eventType == EventType.MouseMove)
                m_GUIWrapper.Repaint();
        }

        public bool CanLayout()
        {
            return m_GUIWrapper.eventType == EventType.Layout;
        }

        public void LayoutBone(int id, Vector3 position, Vector3 endPosition, Vector3 forward, Vector3 up, Vector3 right, bool isChainEnd)
        {
            if (mode == SkeletonMode.Disabled)
                return;

            var sliderData = new SliderData()
            {
                position = GetMouseWorldPosition(forward, position),
                forward = forward,
                up = up,
                right = right
            };

            {
                var distance = m_GUIWrapper.DistanceToSegmentClamp(position, endPosition);

                if (distance <= m_NearestDistance)
                {
                    m_NearestDistance = distance;
                    m_NearestBodyDistance = distance;
                    m_NearestBodyId = id;
                    m_HoveredSliderData = sliderData;
                }
            }

            {
                var distance = m_GUIWrapper.DistanceToCircle(position, GetBoneRadiusForPicking(position) * 2f);

                if (distance <= m_NearestDistance)
                {
                    m_NearestDistance = distance;
                    m_NearestJointDistance = distance;
                    m_NearestJointId = id;
                    m_HoveredSliderData = sliderData;
                }
            }

            if (isChainEnd &&
                (IsCapable(SkeletonAction.ChangeLength) ||
                    IsCapable(SkeletonAction.MoveEndPosition) ||
                    IsCapable(SkeletonAction.CreateBone)))
            {
                var distance = m_GUIWrapper.DistanceToCircle(endPosition, GetBoneRadiusForPicking(endPosition));

                if (distance <= m_NearestDistance)
                {
                    m_NearestDistance = distance;
                    m_NearestTailDistance = distance;
                    m_NearestTailId = id;
                    m_HoveredSliderData = sliderData;
                }
            }
        }

        public Vector3 GetMouseWorldPosition(Vector3 planeNormal, Vector3 planePosition)
        {
            return m_GUIWrapper.GUIToWorld(m_GUIWrapper.mousePosition, planeNormal, planePosition);
        }

        float GetBoneRadiusForPicking(Vector3 position)
        {
            if (m_GUIWrapper.HasCurrentCamera())
                return 0.1f * m_GUIWrapper.GetHandleSize(position);

            return k_PickingRadius;
        }

        public bool DoSelectBone(out int id, out bool additive)
        {
            id = 0;
            additive = false;

            if (IsActionTriggering(SkeletonAction.Select))
            {
                id = m_HoveredBoneID;
                additive = m_GUIWrapper.isActionKeyDown;

                if (mode == SkeletonMode.Selection)
                {
                    m_GUIWrapper.UseCurrentEvent();
                    m_GUIWrapper.SetGuiChanged(true);
                }

                return true;
            }

            return false;
        }

        public bool DoRotateBone(Vector3 pivot, Vector3 normal, out float deltaAngle)
        {
            deltaAngle = 0f;

            var oldPosition = m_HotSliderData.position;
            if (DoSliderAction(SkeletonAction.RotateBone, m_HoveredBodyControlID, ref m_RotateControlID, out var newPosition))
            {
                deltaAngle = Vector3.SignedAngle(oldPosition - pivot, (Vector3)newPosition - pivot, normal);
                return true;
            }

            return false;
        }

        public bool DoMoveBone(out Vector3 deltaPosition)
        {
            deltaPosition = Vector3.zero;

            var oldPosition = m_HotSliderData.position;
            if (DoSliderAction(SkeletonAction.MoveBone, m_HoveredJointControlID, ref m_MoveControlID, out var newPosition))
            {
                deltaPosition = newPosition - oldPosition;
                return true;
            }

            return false;
        }

        public bool DoFreeMoveBone(out Vector3 deltaPosition)
        {
            deltaPosition = Vector3.zero;

            var oldPosition = m_HotSliderData.position;
            if (DoSliderAction(SkeletonAction.FreeMoveBone, m_HoveredBodyControlID, ref m_FreeMoveControlID, out var newPosition))
            {
                deltaPosition = newPosition - oldPosition;
                return true;
            }

            return false;
        }

        public bool DoMoveJoint(out Vector3 deltaPosition)
        {
            deltaPosition = Vector3.zero;

            var oldPosition = m_HotSliderData.position;
            if (DoSliderAction(SkeletonAction.MoveJoint, m_HoveredJointControlID, ref m_MoveJointControlID, out var newPosition))
            {
                deltaPosition = newPosition - oldPosition;
                return true;
            }

            return false;
        }

        public bool DoMoveEndPosition(out Vector3 endPosition)
        {
            return DoSliderAction(SkeletonAction.MoveEndPosition, m_HoveredTailControlID, ref m_MoveEndPositionControlID, out endPosition);
        }

        public bool DoChangeLength(out Vector3 endPosition)
        {
            return DoSliderAction(SkeletonAction.ChangeLength, m_HoveredTailControlID, ref m_ChangeLengthControlID, out endPosition);
        }

        bool DoSliderAction(SkeletonAction action, int controlID, ref int actionControlID, out Vector3 newPosition)
        {
            newPosition = m_HoveredSliderData.position;

            if (IsActionTriggering(action))
            {
                actionControlID = controlID;
                m_HotSliderData = m_HoveredSliderData;
                m_HotBoneID = hoveredBoneID;
            }

            if (m_GUIWrapper.DoSlider(actionControlID, m_HotSliderData, out newPosition))
            {
                m_HotSliderData.position = newPosition;
                return true;
            }

            return false;
        }

        public bool DoCreateBoneStart(out Vector3 position)
        {
            position = GetMouseWorldPosition(m_HoveredSliderData.forward, m_HoveredSliderData.position);

            if (CanCreateBone())
                m_GUIWrapper.LayoutControl(m_CreateBoneControlID, 0f);

            if (IsActionActive(SkeletonAction.CreateBone))
                ConsumeMouseMoveEvents();

            if (IsActionTriggering(SkeletonAction.CreateBone))
            {
                m_HotBoneID = hoveredBoneID;
                m_GUIWrapper.SetMultiStepControlHot(m_CreateBoneControlID);
                m_GUIWrapper.UseCurrentEvent();
                return true;
            }

            return false;
        }

        public bool CanCreateBone()
        {
            return mode == SkeletonMode.CreateBone && (m_GUIWrapper.IsControlNearest(defaultControlID) || m_GUIWrapper.IsControlNearest(m_HoveredTailControlID));
        }

        public bool DoCreateBone(out Vector3 position)
        {
            position = GetMouseWorldPosition(m_HoveredSliderData.forward, m_HoveredSliderData.position);

            if (IsActionHot(SkeletonAction.CreateBone))
                ConsumeMouseMoveEvents();

            if (IsActionFinishing(SkeletonAction.CreateBone))
            {
                m_GUIWrapper.UseCurrentEvent();
                m_GUIWrapper.SetGuiChanged(true);
                return true;
            }

            return false;
        }

        public bool DoSplitBone(out int id, out Vector3 position)
        {
            id = m_HoveredBodyID;
            position = GetMouseWorldPosition(m_HoveredSliderData.forward, m_HoveredSliderData.position);

            if (IsActionActive(SkeletonAction.SplitBone))
                ConsumeMouseMoveEvents();

            if (IsActionTriggering(SkeletonAction.SplitBone))
            {
                m_GUIWrapper.UseCurrentEvent();
                m_GUIWrapper.SetGuiChanged(true);
                return true;
            }

            return false;
        }

        public bool DoRemoveBone()
        {
            if (IsActionTriggering(SkeletonAction.Remove))
            {
                m_GUIWrapper.UseCurrentEvent();
                m_GUIWrapper.SetGuiChanged(true);
                return true;
            }

            return false;
        }

        public bool DoCancelMultistepAction(bool force)
        {
            if (force)
            {
                m_GUIWrapper.SetMultiStepControlHot(0);
                return true;
            }

            if ((!m_GUIWrapper.IsMultiStepControlHot(0) && (m_GUIWrapper.IsMouseDown(1) || m_GUIWrapper.IsKeyDown(KeyCode.Escape))))
            {
                m_GUIWrapper.SetMultiStepControlHot(0);
                m_GUIWrapper.UseCurrentEvent();
                return true;
            }

            return false;
        }

        public bool IsActionActive(SkeletonAction action)
        {
            if (m_GUIWrapper.isAltDown || !m_GUIWrapper.IsControlHot(0) || !m_GUIWrapper.IsMultiStepControlHot(0))
                return false;

            if (action == SkeletonAction.None)
                return m_GUIWrapper.IsControlNearest(defaultControlID);

            if (!IsCapable(action))
                return false;

            if (action == SkeletonAction.RotateBone)
                return m_GUIWrapper.IsControlNearest(m_HoveredBodyControlID);

            if (action == SkeletonAction.ChangeLength)
                return m_GUIWrapper.IsControlNearest(m_HoveredTailControlID) && !m_GUIWrapper.isShiftDown;

            if (action == SkeletonAction.MoveJoint)
                return m_GUIWrapper.IsControlNearest(m_HoveredJointControlID);

            if (action == SkeletonAction.MoveEndPosition)
                return m_GUIWrapper.IsControlNearest(m_HoveredTailControlID) && !m_GUIWrapper.isShiftDown;

            if (action == SkeletonAction.FreeMoveBone)
                return m_GUIWrapper.IsControlNearest(m_HoveredBodyControlID);

            if (action == SkeletonAction.MoveBone)
                return m_GUIWrapper.IsControlNearest(m_HoveredJointControlID);

            bool canCreateBone = IsCapable(SkeletonAction.CreateBone) && m_GUIWrapper.IsControlNearest(m_CreateBoneControlID);
            bool canSplitBone = IsCapable(SkeletonAction.SplitBone) && m_GUIWrapper.IsControlNearest(m_HoveredBodyControlID);

            if (action == SkeletonAction.CreateBone)
                return canCreateBone;

            if (action == SkeletonAction.SplitBone)
                return canSplitBone;

            if (action == SkeletonAction.Select)
                return (m_GUIWrapper.IsControlNearest(m_HoveredBodyControlID) && !canSplitBone) ||
                    m_GUIWrapper.IsControlNearest(m_HoveredJointControlID) ||
                    (m_GUIWrapper.IsControlNearest(m_HoveredTailControlID) && !canCreateBone);

            if (action == SkeletonAction.Remove)
                return true;

            return false;
        }

        public bool IsActionHot(SkeletonAction action)
        {
            if (action == SkeletonAction.None)
                return m_GUIWrapper.IsControlHot(0) && m_GUIWrapper.IsMultiStepControlHot(0);

            if (action == SkeletonAction.RotateBone)
                return m_GUIWrapper.IsControlHot(m_RotateControlID);

            if (action == SkeletonAction.MoveBone)
                return m_GUIWrapper.IsControlHot(m_MoveControlID);

            if (action == SkeletonAction.FreeMoveBone)
                return m_GUIWrapper.IsControlHot(m_FreeMoveControlID);

            if (action == SkeletonAction.MoveJoint)
                return m_GUIWrapper.IsControlHot(m_MoveJointControlID);

            if (action == SkeletonAction.MoveEndPosition)
                return m_GUIWrapper.IsControlHot(m_MoveEndPositionControlID);

            if (action == SkeletonAction.ChangeLength)
                return m_GUIWrapper.IsControlHot(m_ChangeLengthControlID);

            if (action == SkeletonAction.CreateBone)
                return m_GUIWrapper.IsMultiStepControlHot(m_CreateBoneControlID) && !m_GUIWrapper.isAltDown;

            return false;
        }

        public bool IsActionTriggering(SkeletonAction action)
        {
            if (!IsActionActive(action))
                return false;

            if (action == SkeletonAction.Remove)
            {
                if ((m_GUIWrapper.eventType == EventType.ValidateCommand || m_GUIWrapper.eventType == EventType.ExecuteCommand)
                    && (m_GUIWrapper.commandName == softDeleteCommandName || m_GUIWrapper.commandName == deleteCommandName))
                {
                    if (m_GUIWrapper.eventType == EventType.ExecuteCommand)
                        return true;

                    m_GUIWrapper.UseCurrentEvent();
                }

                return false;
            }

            return m_GUIWrapper.IsMouseDown(0);
        }

        public bool IsActionFinishing(SkeletonAction action)
        {
            if (!IsActionHot(action) || !IsCapable(action))
                return false;

            if (m_GUIWrapper.IsEventOutsideWindow())
                return true;

            if (action == SkeletonAction.CreateBone)
                return m_GUIWrapper.IsMouseDown(0);

            return m_GUIWrapper.IsMouseUp(0);
        }

        public bool IsRepainting()
        {
            return m_GUIWrapper.IsRepainting();
        }

        public void DrawBone(Vector3 position, Vector3 right, Vector3 forward, float length, Color color, bool isChained, bool isSelected, bool isJointHovered, bool isTailHovered, bool isHot)
        {
            var endPosition = position + right * length;
            var rotation = Quaternion.LookRotation(forward, Vector3.Cross(right, forward));
            var boneJointColor = new Color(0f, 0f, 0f, 0.75f * color.a);
            var tailColor = new Color(0f, 0f, 0f, 0.75f * color.a);
            var hoveredColor = Handles.preselectionColor;
            var selectedColor = Handles.selectedColor;
            var drawRectCap = false;

            if (isJointHovered)
                boneJointColor = hoveredColor;
            if (isHot && (IsActionHot(SkeletonAction.MoveBone) || IsActionHot(SkeletonAction.MoveJoint)))
                boneJointColor = selectedColor;

            if (mode == SkeletonMode.EditPose || mode == SkeletonMode.CreateBone)
            {
                if (isJointHovered || isSelected)
                    drawRectCap = true;
            }
            else if (mode == SkeletonMode.EditJoints || mode == SkeletonMode.SplitBone)
            {
                rotation = Quaternion.identity;
                drawRectCap = true;
            }

            if (drawRectCap)
                Handles.RectangleHandleCap(0, position, rotation, BoneDrawingUtility.GetBoneRadius(position), EventType.Repaint);

            BoneDrawingUtility.DrawBone(position, endPosition, forward, color);
            BoneDrawingUtility.DrawBoneNode(position, forward, boneJointColor);

            if (!isChained &&
                (IsCapable(SkeletonAction.ChangeLength) ||
                    IsCapable(SkeletonAction.MoveEndPosition)))
            {
                if (isTailHovered)
                    tailColor = hoveredColor;

                if (isHot && (IsActionHot(SkeletonAction.ChangeLength) || IsActionHot(SkeletonAction.MoveEndPosition)))
                    tailColor = selectedColor;

                BoneDrawingUtility.DrawBoneNode(endPosition, forward, tailColor);
            }
        }

        public void DrawBoneParentLink(Vector3 parentPosition, Vector3 position, Vector3 forward, Color color)
        {
            BoneDrawingUtility.DrawBone(position, parentPosition, forward, color);
        }

        public void DrawBoneOutline(Vector3 position, Vector3 right, Vector3 forward, float length, Color color, float outlineScale)
        {
            BoneDrawingUtility.DrawBoneOutline(position, position + right * length, forward, color, outlineScale);
        }

        public void DrawCursors(bool canBeActive)
        {
            var mouseScreenRect = new Rect(m_GUIWrapper.mousePosition.x - 100f, m_GUIWrapper.mousePosition.y - 100f, 200f, 200f);

            var isRotateHot = IsActionHot(SkeletonAction.RotateBone);
            if ((canBeActive && IsActionActive(SkeletonAction.RotateBone)) || isRotateHot)
                EditorGUIUtility.AddCursorRect(mouseScreenRect, MouseCursor.RotateArrow);

            if ((canBeActive && IsActionActive(SkeletonAction.MoveBone)) || IsActionHot(SkeletonAction.MoveBone) ||
                (canBeActive && IsActionActive(SkeletonAction.FreeMoveBone)) || IsActionHot(SkeletonAction.FreeMoveBone) ||
                (canBeActive && IsActionActive(SkeletonAction.MoveJoint)) || IsActionHot(SkeletonAction.MoveJoint) ||
                (canBeActive && IsActionActive(SkeletonAction.MoveEndPosition)) || IsActionHot(SkeletonAction.MoveEndPosition))
                EditorGUIUtility.AddCursorRect(mouseScreenRect, MouseCursor.MoveArrow);
        }

        void ConsumeMouseMoveEvents()
        {
            if (m_GUIWrapper.eventType == EventType.MouseMove || (m_GUIWrapper.eventType == EventType.MouseDrag && m_GUIWrapper.mouseButton == 0))
                m_GUIWrapper.UseCurrentEvent();
        }

        bool IsCapable(SkeletonAction action)
        {
            return ((int)mode & (int)action) != 0;
        }
    }
}
