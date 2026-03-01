import { getPriorityColor, getPriorityLabel, formatDate } from './tabHelpers'

export default function ScheduleTab({ schedule, pushStatus, pushing, onPushToArke }) {
  return (
    <div className="space-y-4">
      {/* Stats + Push button */}
      <div className="flex items-center justify-between mb-2">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1">
          <div className="bg-white rounded-lg shadow p-4 border">
            <div className="text-sm text-gray-500">Total Orders</div>
            <div className="text-2xl font-bold text-gray-900">{schedule?.production_plans?.length || 0}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4 border">
            <div className="text-sm text-gray-500">Policy</div>
            <div className="text-2xl font-bold text-blue-600">EDF</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4 border">
            <div className="text-sm text-gray-500">Conflicts Detected</div>
            <div className="text-2xl font-bold text-red-600">{schedule?.conflicts?.length || 0}</div>
          </div>
          <div className="bg-white rounded-lg shadow p-4 border">
            <div className="text-sm text-gray-500">Pushed to Arke</div>
            <div className="text-2xl font-bold text-green-600">
              {Object.values(pushStatus).filter(r => r.status === 'pushed').length}
              <span className="text-sm font-normal text-gray-400">
                {' / '}{schedule?.production_plans?.length || 0}
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={onPushToArke}
          disabled={pushing}
          className={`ml-4 px-5 py-3 rounded-lg font-semibold text-sm shadow transition flex items-center gap-2 ${
            pushing
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white'
          }`}
        >
          {pushing ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Pushing…
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" />
              </svg>
              Push to Arke
            </>
          )}
        </button>
      </div>

      {/* Production Plans */}
      <div className="space-y-3">
        {schedule?.production_plans?.map((plan, index) => {
          const ps = pushStatus[plan.order_number]
          return (
          <div key={index} className="bg-white rounded-lg shadow-sm border hover:shadow-md transition">
            <div className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-lg font-semibold text-gray-900">#{index + 1}</span>
                    <h3 className="text-lg font-semibold text-gray-900">{plan.order_number}</h3>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getPriorityColor(plan.priority)}`}>
                      P{plan.priority} - {getPriorityLabel(plan.priority)}
                    </span>
                    {/* Push status badge */}
                    {ps?.status === 'pushed' && (
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 border border-green-200">
                        ✓ Pushed
                      </span>
                    )}
                    {ps?.status === 'failed' && (
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700 border border-red-200" title={ps.error_message}>
                        ✗ Failed
                      </span>
                    )}
                    {ps?.status === 'pushing' && (
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 border border-yellow-200 animate-pulse">
                        ⏳ Pushing…
                      </span>
                    )}
                    {!ps && (
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-500 border border-gray-200">
                        ○ Pending
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Product:</span>
                      <p className="font-medium text-gray-900">{plan.product}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Customer:</span>
                      <p className="font-medium text-gray-900">{plan.customer}</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Quantity:</span>
                      <p className="font-medium text-gray-900">{plan.quantity} units</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Schedule:</span>
                      <p className="font-medium text-gray-900">
                        {formatDate(plan.starts_at)} → {formatDate(plan.ends_at)}
                      </p>
                    </div>
                  </div>
                  {plan.reasoning && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-md">
                      <p className="text-sm text-gray-700">
                        <span className="font-medium">Reasoning:</span> {plan.reasoning}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          )
        })}
      </div>
    </div>
  )
}
