//! Facilitates establishing relationships between entities in the DB,
//! allowing reusing the same type as both primary and foreign key
//! without cross-references between entity modules.

use spacetimedb::{Identity, SpacetimeType};
use strum::Display;

#[derive(SpacetimeType, Clone)]
pub enum ActorId {
	Internal(AccountId),
	External(ExternalActorId),
}

/// Primary key for the account table
pub type AccountId = Identity;

/// Primary key for the external actor table
///
/// Must convey the following format:
/// `"{String}@{ExternalActorOrigin}"`
pub type ExternalActorId = String;

#[derive(SpacetimeType, Clone, Display)]
pub enum ChannelId {
	Direct(DirectChannelId),
	Standalone(StandaloneChannelId),
	Primary(PrimaryChannelId),
	Subordinate(SubordinateChannelId),
}

pub type DirectChannelId = String;

pub type StandaloneChannelId = String;

pub type PrimaryChannelId = String;

pub type SubordinateChannelId = String;
