using System;
using PhotoshopFile;
using UnityEngine;
using UnityEngine.Analytics;

namespace UnityEditor.U2D.PSD
{
    [Serializable]
    internal struct PSDApplyEvent
#if USE_NEW_EDITOR_ANALYTICS
        : IAnalytic.IData
#endif
    {
        public const string name = "psdImporterApply";

        public int instance_id;
        public int texture_type;
        public int sprite_mode;
        public bool mosaic_layer;
        public bool import_hidden_layer;
        public bool character_mode;
        public bool generate_go_hierarchy;
        public bool reslice_from_layer;
        public bool is_character_rigged;
        public SpriteAlignment character_alignment;
        public bool is_psd;
        public PsdColorMode color_mode;
    }

#if USE_NEW_EDITOR_ANALYTICS
    [AnalyticInfo(eventName: PSDApplyEvent.name,
        vendorKey: Analytics.vendorKey,
        version: Analytics.version,
        maxEventsPerHour: Analytics.maxEventsPerHour,
        maxNumberOfElements: Analytics.maxNumberOfElements)]
    internal class PSDApplyEventAnalytic : IAnalytic
    {
        PSDApplyEvent m_EvtData;

        public PSDApplyEventAnalytic(PSDApplyEvent evtData)
        {
            m_EvtData = evtData;
        }

        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            data = m_EvtData;
            error = null;
            return true;
        }
    }
#endif
    
    internal interface IAnalytics
    {
        AnalyticsResult SendApplyEvent(PSDApplyEvent evt);
    }

    internal static class AnalyticFactory
    {
        static IAnalytics s_Analytics;

        public static IAnalytics analytics
        {
            get => s_Analytics ??= new Analytics();
            set => s_Analytics = value;
        }
    }

    [InitializeOnLoad]
    internal class Analytics : IAnalytics
    {
        public const int maxEventsPerHour = 100;
        public const int maxNumberOfElements = 1000;
        public const string vendorKey = "unity.2d.psdimporter";
        public const int version = 1;

        static Analytics()
        {
#if !USE_NEW_EDITOR_ANALYTICS
            EditorAnalytics.RegisterEventWithLimit(PSDApplyEvent.name, maxEventsPerHour, maxNumberOfElements, vendorKey, version);
#endif
        }

        public AnalyticsResult SendApplyEvent(PSDApplyEvent evt)
        {
#if USE_NEW_EDITOR_ANALYTICS
            return EditorAnalytics.SendAnalytic(new PSDApplyEventAnalytic(evt));
#else
            return EditorAnalytics.SendEventWithLimit(PSDApplyEvent.name, evt, version);
#endif
        }
    }
}