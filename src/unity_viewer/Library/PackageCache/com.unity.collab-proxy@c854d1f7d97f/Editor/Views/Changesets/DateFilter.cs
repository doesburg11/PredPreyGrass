using PlasticGui.WorkspaceWindow.QueryViews;

using LayoutFilters = Codice.Client.BaseCommands.LayoutFilters;

namespace Unity.PlasticSCM.Editor.Views.Changesets
{
    internal class DateFilter
    {
        internal enum Type
        {
            LastWeek,
            Last15Days,
            LastMonth,
            Last3Months,
            LastYear,
            AllTime
        }

        internal Type FilterType;

        internal DateFilter(Type filterType)
        {
            FilterType = filterType;
        }

        internal LayoutFilters.DateFilter GetLayoutFilter()
        {
            switch (FilterType)
            {
                case Type.LastWeek:
                    return LayoutFilters.DateFilter.BuildFromTimeAgo(
                        LayoutFilters.SinceTimeType.OneWeekAgo);
                case Type.Last15Days:
                    return LayoutFilters.DateFilter.BuildFromTimeAgo(
                        LayoutFilters.SinceTimeType.FifteenDaysAgo);
                case Type.LastMonth:
                    return LayoutFilters.DateFilter.BuildFromTimeAgo(
                        LayoutFilters.SinceTimeType.OneMonthAgo);
                case Type.Last3Months:
                    return LayoutFilters.DateFilter.BuildFromTimeAgo(
                        LayoutFilters.SinceTimeType.ThreeMonthsAgo);
                case Type.LastYear:
                    return LayoutFilters.DateFilter.BuildFromTimeAgo(
                        LayoutFilters.SinceTimeType.OneYearAgo);
            }

            return null;
        }

        internal string GetTimeAgo()
        {
            switch (FilterType)
            {
                case Type.LastWeek:
                    return QueryConstants.OneWeekAgo;
                case Type.Last15Days:
                    return QueryConstants.HalfMonthAgo;
                case Type.LastMonth:
                    return QueryConstants.OneMonthAgo;
                case Type.Last3Months:
                    return QueryConstants.ThreeMonthsAgo;
                case Type.LastYear:
                    return QueryConstants.OneYearAgo;
            }

            return string.Empty;
        }
    }
}
