use crate::algo::token::*;
use crate::interp::*;
use crate::panick::{_debugln, _traceln};

use std::any::{Any};

#[derive(Debug, Default)]
pub struct ChoiceFun {
  // NB: this is just a "function"!
}

impl Function for ChoiceFun {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn __apply__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<Option<Yield_>, InterpCheck> {
    let clk = interp.clkctr._get_clock();
    let nextclk = interp.clkctr._next_clock();
    let xlb = interp.reg.xlb;
    let rst_clk = interp.reg.rst_clk;
    /*if rst_clk.is_nil() {
      return Err(bot());
    }*/
    // NB: need to nil out reg.rst_clk.
    interp.reg.rst_clk = nil();
    _traceln!(interp, "DEBUG: ChoiceFun::__apply__: clk={:?} nextclk={:?} xlb={:?} tup.len={}", clk, nextclk, xlb, tup.len());

    if tup.len() > 2 {
      _traceln!(interp, "DEBUG: ChoiceFun::__apply__:   tup={:?}", tup);
      return Err(bot());
    }

    let mut choice_ub: Option<RawChoiceRank> = None;
    if tup.len() > 1 {
      _traceln!(interp, "DEBUG: ChoiceFun::__apply__:   tup={:?}", tup);
      let vals = interp.get_vals(clk, tup[1])?;
      _traceln!(interp, "DEBUG: ChoiceFun::__apply__:   vals={:?}", vals);
      for &(key, ref val) in vals.iter() {
        match val {
          &LitVal_::Int(v) => {
            choice_ub = Some(v.try_into().unwrap());
            // TODO: when to catch contradictory vals?
            break;
          }
          _ => {}
        }
      }
    }

    // Need to distinguish b/w the two cases:
    // - this is the initial choice() (i.e. we are on the PV)
    // - this is a later choice() due to failure/backtracking
    //
    // Q: how to identify a choice point code block?
    // - static lexical info: reference this block via `this_span`
    // - semi-static code num info
    //   - not necessarily static, in the case of quote-based codegen
    // - dynamic linear timestamp
    //   - so, we have to leave the trace entry of _this choice point_ in place
    //   - let's do this version
    //
    // (in the old implementation, this was a non-problem b/c backtracking
    // always replayed the whole log/choice trace from the start.)
    let te = if !rst_clk.is_nil() {
      match interp.trace._maybe_get(rst_clk) {
        Some(te) => {
          _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace: get: rst clk={:?} clk={:?}", rst_clk, clk);
          te.last_clk.set(clk);
          te
        }
        None => {
          let ctl_reg = FastCtlReg_{
            exc_: interp.exc_.clone(),
            res_: interp.res_.clone(),
            port: interp.port.clone(),
          };
          let knt_ = MemKnt{
            clk:  knt.clk,
            prev: knt.prev.clone(),
            cur:  knt.cur,
          }.into_ref();
          _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace: push: clk={:?}", clk);
          interp.trace._push(clk, choice_ub.unwrap_or(u16::max_value()), xlb, interp.reg, ctl_reg, knt_)?;
          interp.trace._maybe_get(clk).unwrap()
        }
      }
    } else {
      match interp.trace._maybe_get(rst_clk) {
        Some(_) => {
          _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace: get: rst clk={:?} clk={:?}", rst_clk, clk);
          return Err(bot());
        }
        None => {
          let ctl_reg = FastCtlReg_{
            exc_: interp.exc_.clone(),
            res_: interp.res_.clone(),
            port: interp.port.clone(),
          };
          let knt_ = MemKnt{
            clk:  knt.clk,
            prev: knt.prev.clone(),
            cur:  knt.cur,
          }.into_ref();
          _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace: push: clk={:?}", clk);
          interp.trace._push(clk, choice_ub.unwrap_or(u16::max_value()), xlb, interp.reg, ctl_reg, knt_)?;
          interp.trace._maybe_get(clk).unwrap()
        }
      }
    };
    _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace.buf.len={}", &interp.trace.buf.len());
    _traceln!(interp, "DEBUG: ChoiceFun::__apply__: log.buf.len  ={}", &interp.log.buf.len());
    _traceln!(interp, "DEBUG: ChoiceFun::__apply__: trace.buf={:?}", &interp.trace.buf);
    _traceln!(interp, "DEBUG: ChoiceFun::__apply__: log.buf  ={:?}", &interp.log.buf);

    // NB: the choice point counter should be exposed as a val.
    let choice_ctr = te.xctr;
    if choice_ub.is_none() || choice_ctr < choice_ub.unwrap() {
      _traceln!(interp, "DEBUG: ChoiceFun::__apply__: choice ctr={} ub={:?}", choice_ctr, choice_ub);
      let y = interp._fresh().into_term();
      let val_ = LitVal_::Int(choice_ctr.into());
      interp.put_val(clk, y, val_)?;
      interp.unify(clk, y, ret)?;
      Ok(None)
    } else {
      _traceln!(interp, "DEBUG: ChoiceFun::__apply__: choice ctr={} ub={:?} fail", choice_ctr, choice_ub);
      Ok(Some(Yield_::Fail))
    }
  }
}

#[derive(Debug, Default)]
pub struct FailureFun {
  // NB: this is just a "function"!
}

impl Function for FailureFun {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn __apply__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<Option<Yield_>, InterpCheck> {
    Ok(Some(Yield_::Fail))
  }
}

#[derive(Debug, Default)]
pub struct EvalFun {
}

impl Function for EvalFun {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn __apply__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<Option<Yield_>, InterpCheck> {
    let clk = interp.clkctr._get_clock();
    let nextclk = interp.clkctr._next_clock();
    let xlb = interp.reg.xlb;
    _traceln!(interp, "DEBUG: EvalFun::__apply__: clk={:?} nextclk={:?} xlb={:?} tup.len={}", clk, nextclk, xlb, tup.len());

    if tup.len() != 1 {
      _traceln!(interp, "DEBUG: EvalFun::__apply__:   tup={:?}", tup);
      return Err(bot());
    }

    {
      _traceln!(interp, "DEBUG: EvalFun::__apply__:   tup={:?}", tup);
      // TODO
      return Err(unimpl());
    }

    Ok(Some(Yield_::Eval))
  }
}

#[derive(Debug, Default)]
pub struct PrintFun {
}

impl Function for PrintFun {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn __apply__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<Option<Yield_>, InterpCheck> {
    let clk = interp.clkctr._get_clock();
    let nextclk = interp.clkctr._next_clock();
    let xlb = interp.reg.xlb;
    _traceln!(interp, "DEBUG: PrintFun::__apply__: clk={:?} nextclk={:?} xlb={:?} tup.len={}", clk, nextclk, xlb, tup.len());

    if tup.len() > 2 {
      _traceln!(interp, "DEBUG: PrintFun::__apply__:   tup={:?}", tup);
      return Err(bot());
    }

    if tup.len() > 1 {
      _traceln!(interp, "DEBUG: PrintFun::__apply__:   tup={:?}", tup);
      let vals = interp.get_vals(clk, tup[1])?;
      _traceln!(interp, "DEBUG: PrintFun::__apply__:   vals={:?}", vals);
      for &(key, ref val) in vals.iter() {
        match val {
          &LitVal_::None => {
            println!("None");
          }
          &LitVal_::Bool(v) => {
            println!("{}", v);
          }
          &LitVal_::Int(v) => {
            println!("{}", v);
          }
          &LitVal_::Atom(ref v) => {
            println!("{}", v);
          }
          _ => {
            // TODO
          }
        }
        // TODO: when to catch contradictory vals?
        break;
      }
    }

    Ok(None)
  }
}

#[derive(Debug, Default)]
pub struct TokenTrieCls {
  // TODO
}

impl ObjectCls for TokenTrieCls {
  fn as_any(&self) -> &dyn Any {
    self
  }

  fn __create__(&mut self, interp: &mut FastInterp, this_span: SpanNum, this_term: SNum, tup: &[ENum], ret: SNum, knt: BorrowedMemKnt, ) -> Result<(), InterpCheck> {
    // TODO
    unimplemented!();
  }
}

#[derive(Debug, Default)]
pub struct TokenTrieVal {
  // TODO
}
