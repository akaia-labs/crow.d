// THIS FILE IS AUTOMATICALLY GENERATED BY SPACETIMEDB. EDITS TO THIS FILE
// WILL NOT BE SAVED. MODIFY TABLES IN YOUR MODULE SOURCE CODE INSTEAD.

#![allow(unused, clippy::all)]
use spacetimedb_sdk::__codegen::{self as __sdk, __lib, __sats, __ws};

use super::tp_account_reference_type::TpAccountReference;

#[derive(__lib::ser::Serialize, __lib::de::Deserialize, Clone, PartialEq, Debug)]
#[sats(crate = __lib)]
pub(super) struct ImportMessageArgs {
	pub author_reference: TpAccountReference,
	pub text:             String,
}

impl From<ImportMessageArgs> for super::Reducer {
	fn from(args: ImportMessageArgs) -> Self {
		Self::ImportMessage {
			author_reference: args.author_reference,
			text:             args.text,
		}
	}
}

impl __sdk::InModule for ImportMessageArgs {
	type Module = super::RemoteModule;
}

pub struct ImportMessageCallbackId(__sdk::CallbackId);

#[allow(non_camel_case_types)]
/// Extension trait for access to the reducer `import_message`.
///
/// Implemented for [`super::RemoteReducers`].
pub trait import_message {
	/// Request that the remote module invoke the reducer `import_message` to
	/// run as soon as possible.
	///
	/// This method returns immediately, and errors only if we are unable to
	/// send the request. The reducer will run asynchronously in the future,
	///  and its status can be observed by listening for
	/// [`Self::on_import_message`] callbacks.
	fn import_message(
		&self, author_reference: TpAccountReference, text: String,
	) -> __sdk::Result<()>;
	/// Register a callback to run whenever we are notified of an invocation of
	/// the reducer `import_message`.
	///
	/// Callbacks should inspect the [`__sdk::ReducerEvent`] contained in the
	/// [`super::ReducerEventContext`] to determine the reducer's status.
	///
	/// The returned [`ImportMessageCallbackId`] can be passed to
	/// [`Self::remove_on_import_message`] to cancel the callback.
	fn on_import_message(
		&self,
		callback: impl FnMut(&super::ReducerEventContext, &TpAccountReference, &String) + Send + 'static,
	) -> ImportMessageCallbackId;
	/// Cancel a callback previously registered by [`Self::on_import_message`],
	/// causing it not to run in the future.
	fn remove_on_import_message(&self, callback: ImportMessageCallbackId);
}

impl import_message for super::RemoteReducers {
	fn import_message(
		&self, author_reference: TpAccountReference, text: String,
	) -> __sdk::Result<()> {
		self.imp.call_reducer("import_message", ImportMessageArgs {
			author_reference,
			text,
		})
	}

	fn on_import_message(
		&self,
		mut callback: impl FnMut(&super::ReducerEventContext, &TpAccountReference, &String)
		+ Send
		+ 'static,
	) -> ImportMessageCallbackId {
		ImportMessageCallbackId(self.imp.on_reducer(
			"import_message",
			Box::new(move |ctx: &super::ReducerEventContext| {
				let super::ReducerEventContext {
					event:
						__sdk::ReducerEvent {
							reducer:
								super::Reducer::ImportMessage {
									author_reference,
									text,
								},
							..
						},
					..
				} = ctx
				else {
					unreachable!()
				};
				callback(ctx, author_reference, text)
			}),
		))
	}

	fn remove_on_import_message(&self, callback: ImportMessageCallbackId) {
		self.imp.remove_on_reducer("import_message", callback.0)
	}
}

#[allow(non_camel_case_types)]
#[doc(hidden)]
/// Extension trait for setting the call-flags for the reducer `import_message`.
///
/// Implemented for [`super::SetReducerFlags`].
///
/// This type is currently unstable and may be removed without a major version
/// bump.
pub trait set_flags_for_import_message {
	/// Set the call-reducer flags for the reducer `import_message` to `flags`.
	///
	/// This type is currently unstable and may be removed without a major
	/// version bump.
	fn import_message(&self, flags: __ws::CallReducerFlags);
}

impl set_flags_for_import_message for super::SetReducerFlags {
	fn import_message(&self, flags: __ws::CallReducerFlags) {
		self.imp.set_call_reducer_flags("import_message", flags);
	}
}
