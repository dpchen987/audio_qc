# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: paraformer.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10paraformer.proto\x12\nparaformer\"^\n\x07Request\x12\x12\n\naudio_data\x18\x01 \x01(\x0c\x12\x0c\n\x04user\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x10\n\x08speaking\x18\x04 \x01(\x08\x12\r\n\x05isEnd\x18\x05 \x01(\x08\"L\n\x08Response\x12\x10\n\x08sentence\x18\x01 \x01(\t\x12\x0c\n\x04user\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x04 \x01(\t2C\n\x03\x41SR\x12<\n\tRecognize\x12\x13.paraformer.Request\x1a\x14.paraformer.Response\"\x00(\x01\x30\x01\x42\x16\n\x07\x65x.grpc\xa2\x02\nparaformerb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'paraformer_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\007ex.grpc\242\002\nparaformer'
  _REQUEST._serialized_start=32
  _REQUEST._serialized_end=126
  _RESPONSE._serialized_start=128
  _RESPONSE._serialized_end=204
  _ASR._serialized_start=206
  _ASR._serialized_end=273
# @@protoc_insertion_point(module_scope)
