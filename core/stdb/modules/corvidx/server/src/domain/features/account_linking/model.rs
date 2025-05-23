use spacetimedb::{Identity, ScheduleAt, Timestamp, table};

use super::reducers::scheduled_delete_account_link_request;
use crate::domain::entities::{native_account::NativeAccountId, tp_account::TpAccountId};

pub type AccountLinkRequestId = u64;

#[table(name = account_link_request, public)]
/// Represents a pending link request
/// from a native account to a third-party account
pub struct AccountLinkRequest {
	#[primary_key]
	#[auto_inc]
	pub id: AccountLinkRequestId,

	pub issuer:               Identity,
	pub created_at:           Timestamp,
	pub expires_at:           Timestamp,
	pub requester_account_id: NativeAccountId,
	pub subject_account_id:   TpAccountId,
}

#[table(name = account_link_request_schedule, scheduled(scheduled_delete_account_link_request))]
pub struct AccountLinkRequestExpirySchedule {
	#[primary_key]
	#[auto_inc]
	pub scheduled_id: u64,

	pub scheduled_at: ScheduleAt,
	pub request_id:   AccountLinkRequestId,
}
