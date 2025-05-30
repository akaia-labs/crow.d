// THIS FILE IS AUTOMATICALLY GENERATED BY SPACETIMEDB. EDITS TO THIS FILE
// WILL NOT BE SAVED. MODIFY TABLES IN YOUR MODULE SOURCE CODE INSTEAD.

#![allow(unused, clippy::all)]
use spacetimedb_sdk::__codegen::{self as __sdk, __lib, __sats, __ws};

use super::account_link_request_expiry_schedule_type::AccountLinkRequestExpirySchedule;

#[derive(__lib::ser::Serialize, __lib::de::Deserialize, Clone, PartialEq, Debug)]
#[sats(crate = __lib)]
pub(super) struct ScheduledDeleteAccountLinkRequestArgs {
	pub args: AccountLinkRequestExpirySchedule,
}

impl From<ScheduledDeleteAccountLinkRequestArgs> for super::Reducer {
	fn from(args: ScheduledDeleteAccountLinkRequestArgs) -> Self {
		Self::ScheduledDeleteAccountLinkRequest { args: args.args }
	}
}

impl __sdk::InModule for ScheduledDeleteAccountLinkRequestArgs {
	type Module = super::RemoteModule;
}

pub struct ScheduledDeleteAccountLinkRequestCallbackId(__sdk::CallbackId);

#[allow(non_camel_case_types)]
/// Extension trait for access to the reducer
/// `scheduled_delete_account_link_request`.
///
/// Implemented for [`super::RemoteReducers`].
pub trait scheduled_delete_account_link_request {
	/// Request that the remote module invoke the reducer
	/// `scheduled_delete_account_link_request` to run as soon as possible.
	///
	/// This method returns immediately, and errors only if we are unable to
	/// send the request. The reducer will run asynchronously in the future,
	///  and its status can be observed by listening for
	/// [`Self::on_scheduled_delete_account_link_request`] callbacks.
	fn scheduled_delete_account_link_request(
		&self, args: AccountLinkRequestExpirySchedule,
	) -> __sdk::Result<()>;
	/// Register a callback to run whenever we are notified of an invocation of
	/// the reducer `scheduled_delete_account_link_request`.
	///
	/// Callbacks should inspect the [`__sdk::ReducerEvent`] contained in the
	/// [`super::ReducerEventContext`] to determine the reducer's status.
	///
	/// The returned [`ScheduledDeleteAccountLinkRequestCallbackId`] can be
	/// passed to [`Self::remove_on_scheduled_delete_account_link_request`]
	/// to cancel the callback.
	fn on_scheduled_delete_account_link_request(
		&self,
		callback: impl FnMut(&super::ReducerEventContext, &AccountLinkRequestExpirySchedule)
		+ Send
		+ 'static,
	) -> ScheduledDeleteAccountLinkRequestCallbackId;
	/// Cancel a callback previously registered by
	/// [`Self::on_scheduled_delete_account_link_request`], causing it not to
	/// run in the future.
	fn remove_on_scheduled_delete_account_link_request(
		&self, callback: ScheduledDeleteAccountLinkRequestCallbackId,
	);
}

impl scheduled_delete_account_link_request for super::RemoteReducers {
	fn scheduled_delete_account_link_request(
		&self, args: AccountLinkRequestExpirySchedule,
	) -> __sdk::Result<()> {
		self.imp.call_reducer(
			"scheduled_delete_account_link_request",
			ScheduledDeleteAccountLinkRequestArgs { args },
		)
	}

	fn on_scheduled_delete_account_link_request(
		&self,
		mut callback: impl FnMut(&super::ReducerEventContext, &AccountLinkRequestExpirySchedule)
		+ Send
		+ 'static,
	) -> ScheduledDeleteAccountLinkRequestCallbackId {
		ScheduledDeleteAccountLinkRequestCallbackId(self.imp.on_reducer(
			"scheduled_delete_account_link_request",
			Box::new(move |ctx: &super::ReducerEventContext| {
				let super::ReducerEventContext {
					event:
						__sdk::ReducerEvent {
							reducer: super::Reducer::ScheduledDeleteAccountLinkRequest { args },
							..
						},
					..
				} = ctx
				else {
					unreachable!()
				};
				callback(ctx, args)
			}),
		))
	}

	fn remove_on_scheduled_delete_account_link_request(
		&self, callback: ScheduledDeleteAccountLinkRequestCallbackId,
	) {
		self.imp
			.remove_on_reducer("scheduled_delete_account_link_request", callback.0)
	}
}

#[allow(non_camel_case_types)]
#[doc(hidden)]
/// Extension trait for setting the call-flags for the reducer
/// `scheduled_delete_account_link_request`.
///
/// Implemented for [`super::SetReducerFlags`].
///
/// This type is currently unstable and may be removed without a major version
/// bump.
pub trait set_flags_for_scheduled_delete_account_link_request {
	/// Set the call-reducer flags for the reducer
	/// `scheduled_delete_account_link_request` to `flags`.
	///
	/// This type is currently unstable and may be removed without a major
	/// version bump.
	fn scheduled_delete_account_link_request(&self, flags: __ws::CallReducerFlags);
}

impl set_flags_for_scheduled_delete_account_link_request for super::SetReducerFlags {
	fn scheduled_delete_account_link_request(&self, flags: __ws::CallReducerFlags) {
		self.imp
			.set_call_reducer_flags("scheduled_delete_account_link_request", flags);
	}
}
