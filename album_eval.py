__author__ = 'Licheng'
from pprint import pprint
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider

class AlbumEvaluator:
	def __init__(self, vist_sis, preds):
		"""
		:params vist_sis: vist's Story_in_Sequence instance
		:params preds   : [{'album_id', 'pred_story_str'}]
		"""
		self.vist_sis = vist_sis
		self.eval_overall = {}   	# overall score
		self.eval_albums  = {}      # score on each album_id
		self.album_to_eval = {} 	# album_id -> eval
		self.preds = preds 			# [{album_id, pred_story_str}]

	def evaluate(self, measure=None):
		"""
		measure is a subset of ['bleu', 'meteor', 'rouge', 'cider']
		if measure is None, we will apply all the above.
		"""

		# album_id -> pred story str
		album_to_Res = {item['album_id']: [item['pred_story_str'].encode('ascii', 'ignore').decode('ascii')]
						for item in self.preds }

		# album_id -> gt story str(s)
		album_to_Gts = {}
		for album_id in album_to_Res.keys():
			album = self.vist_sis.Albums[album_id]
			gd_story_strs = []
			for story_id in album['story_ids']:
				gd_sent_ids = self.vist_sis.Stories[story_id]['sent_ids']
				gd_story_str = ' '.join([self.vist_sis.Sents[sent_id]['text'] for sent_id in gd_sent_ids])
				gd_story_str = gd_story_str.encode('ascii', 'ignore').decode('ascii')  # ignore some weird token
				gd_story_strs += [gd_story_str]
			album_to_Gts[album_id] = gd_story_strs

		self.album_to_Res = album_to_Res
		self.album_to_Gts = album_to_Gts

		# =================================================
		# Set up scorers
		# =================================================
		print 'setting up scorers...'
		scorers = []
		if not measure:
			scorers = [
				(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
				(Meteor(),"METEOR"),
				(Rouge(), "ROUGE_L"),
				(Cider(), "CIDEr")
			]
		else:
			if 'bleu' in measure:
				scorers += [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
			if 'meteor' in measure:
				scorers += [(Meteor(),"METEOR")]
			if 'rouge' in measure:
				scorers += [(Rouge(), "ROUGE_L")]
			if 'cider' in measure:
				scorers += [(Cider(), "CIDEr")]

		# =================================================
		# Compute scores
		# =================================================
		for scorer, method in scorers:
			print 'computing %s score ...' % (scorer.method())
			score, scores = scorer.compute_score(self.album_to_Gts, self.album_to_Res)
			if type(method) == list:
				for sc, scs, m in zip(score, scores, method):
					self.setEval(sc, m)
					self.setAlbumToEval(scs, self.album_to_Gts.keys(), m)
					print '%s: %.3f' % (m, sc)
			else:
				self.setEval(score, method)
				self.setAlbumToEval(scores, self.album_to_Gts.keys(), method)
				print '%s: %.3f' % (method, score)

		self.setEvalAlbums()

	def setEval(self, score, method):
		self.eval_overall[method] = score

	def setAlbumToEval(self, scores, album_ids, method):
		for album_id, score in zip(album_ids, scores):
			if not album_id in self.album_to_eval:
				self.album_to_eval[album_id] = {}
				self.album_to_eval[album_id]['album_id'] = album_id
			self.album_to_eval[album_id][method] = score

	def setEvalAlbums(self):
		self.eval_albums = [eval for album_id, eval in self.album_to_eval.items()]


if __name__ == '__main__':

	import os.path as osp
	from pprint import pprint
	import sys
	sys.path.insert(0, '../vist_api')
	from vist import Story_in_Sequence
	sis = Story_in_Sequence('/playpen/data/vist/images256', '/playpen/data/vist/annotations')

	# fake a predictions
	preds = []
	for album in sis.albums[:1]:
		story_id = album['story_ids'][0]
		story = sis.Stories[story_id]
		pred_story_str = ' '.join([sis.Sents[sent_id]['text'] for sent_id in story['sent_ids']])
		preds += [{'album_id': album['id'], 'pred_story_str': pred_story_str}]

	# load AlbumEval
	aeval = AlbumEvaluator(sis, preds)
	aeval.evaluate(['meteor'])
