use spacetimedb::{Identity, SpacetimeType, Timestamp, table};

use crate::domain::entities::{native_account::NativeAccountId, tp_account::TpAccountId};

#[derive(SpacetimeType)]
/// The original message author.
pub enum MessageAuthorId {
	NativeAccountId(NativeAccountId),
	TpAccountId(TpAccountId),
	/// Fallback value, use with caution.
	Unknown,
}

#[table(name = message, public)]
pub struct Message {
	#[auto_inc]
	#[primary_key]
	pub id: i128,

	pub sent_at: Timestamp,
	pub sender:  Identity,

	#[index(btree)]
	pub author_id: MessageAuthorId,

	pub text: String,
	// TODO: track message forwarding
	// pub forwarded_to: Vec<TpChannelId>
}
